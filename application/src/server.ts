import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { and, desc, eq, inArray } from "drizzle-orm";
import { db, schema } from "./db";
import type { Color, GameStatus, PieceType } from "./db/schema";
import {
  getCol,
  getGameStatus,
  getRow,
  getSquareFromRowCol,
  getValidMoves,
  initializeGame,
  isLegalMove,
  piecesToFen,
} from "./lib/chess/index";

const app = new Hono();

// ─── API routes ───────────────────────────────────────────────────────────────

app.post("/api/board", async (c) => {
  const serverTime = Date.now();

  let currentGame = await db.query.games.findFirst({
    where: eq(schema.games.mode, "vs_computer"),
    orderBy: desc(schema.games.updatedAt),
  });

  if (!currentGame) {
    const initialData = initializeGame();
    const [newGame] = await db
      .insert(schema.games)
      .values({
        currentTurn: initialData.turn,
        status: "Ongoing",
        mode: "vs_computer",
        timeControl: 0,
        whiteTimeRemaining: Number.MAX_SAFE_INTEGER,
        blackTimeRemaining: Number.MAX_SAFE_INTEGER,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      })
      .returning();
    if (newGame) {
      await db.insert(schema.pieces).values(
        initialData.pieces.map((p) => ({
          gameId: newGame.id,
          color: p.color,
          pieceType: p.pieceType,
          square: p.square,
          hasMoved: false,
        })),
      );
      currentGame = newGame;
    }
  }

  if (!currentGame) {
    return c.json({
      id: 0,
      pieces: [],
      turn: "White" as Color,
      status: "Ongoing" as GameStatus,
      moves: [],
      capturedPieces: { white: [], black: [] },
      lastMove: null,
      serverTime,
    });
  }

  const pieces = await db.query.pieces.findMany({
    where: eq(schema.pieces.gameId, currentGame.id),
  });

  const moveHistory = await db.query.moves.findMany({
    where: eq(schema.moves.gameId, currentGame.id),
    orderBy: schema.moves.moveNumber,
  });

  const capturedPieces: { white: PieceType[]; black: PieceType[] } = { white: [], black: [] };

  const formattedMoves = moveHistory.map((move) => {
    const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const fromFile = files[getCol(move.fromSquare)];
    const fromRank = 8 - getRow(move.fromSquare);
    const toFile = files[getCol(move.toSquare)];
    const toRank = 8 - getRow(move.toSquare);
    let notation = "";
    if (move.pieceType !== "Pawn") {
      notation += move.pieceType === "Knight" ? "N" : move.pieceType[0];
    }
    notation += `${fromFile}${fromRank}-${toFile}${toRank}`;
    if (move.capturedPieceType) {
      if (move.pieceColor === "White") capturedPieces.black.push(move.capturedPieceType);
      else capturedPieces.white.push(move.capturedPieceType);
    }
    return {
      from: move.fromSquare,
      to: move.toSquare,
      color: move.pieceColor,
      pieceType: move.pieceType,
      captured: move.capturedPieceType || undefined,
      notation,
    };
  });

  const lastMove =
    formattedMoves.length > 0
      ? { from: formattedMoves[formattedMoves.length - 1].from, to: formattedMoves[formattedMoves.length - 1].to }
      : null;

  return c.json({
    id: currentGame.id,
    pieces: pieces.map((p) => ({ color: p.color, piece_type: p.pieceType, square: p.square })),
    capturedPieces,
    moves: formattedMoves,
    turn: currentGame.currentTurn,
    status: currentGame.status,
    lastMove,
    serverTime,
  });
});

app.post("/api/moves", async (c) => {
  const { square, gameId } = await c.req.json<{ square: number; gameId: number }>();
  const currentGame = await db.query.games.findFirst({ where: eq(schema.games.id, gameId) });
  if (!currentGame) return c.json([]);
  const pieces = await db.query.pieces.findMany({ where: eq(schema.pieces.gameId, currentGame.id) });
  const lastMove = await db.query.moves.findFirst({
    where: eq(schema.moves.gameId, currentGame.id),
    orderBy: desc(schema.moves.moveNumber),
  });
  return c.json(getValidMoves(pieces, square, lastMove ?? undefined));
});

app.post("/api/make-move", async (c) => {
  const { from, to, gameId, engineType } = await c.req.json<{
    from: number; to: number; gameId: number; engineType: "minimax" | "neural";
  }>();

  const currentGame = await db.query.games.findFirst({ where: eq(schema.games.id, gameId) });
  if (!currentGame) return c.json({ error: "No game found" }, 404);
  if (currentGame.status !== "Ongoing") return c.json({ error: "Game is not ongoing" }, 400);

  const pieces = await db.query.pieces.findMany({ where: eq(schema.pieces.gameId, currentGame.id) });
  const lastMove = await db.query.moves.findFirst({
    where: eq(schema.moves.gameId, currentGame.id),
    orderBy: desc(schema.moves.moveNumber),
  });

  if (!isLegalMove(pieces, from, to, currentGame.currentTurn, lastMove ?? undefined)) {
    return c.json({ error: "Invalid move" }, 400);
  }

  const now = Date.now();
  const movingPiece = pieces.find((p) => p.square === from);
  if (!movingPiece) return c.json({ error: "Piece not found" }, 400);

  const capturedPiece = pieces.find((p) => p.square === to);
  let capturedPieceType: PieceType | undefined;

  if (capturedPiece) {
    capturedPieceType = capturedPiece.pieceType;
    await db.delete(schema.pieces).where(and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, to)));
  } else if (movingPiece.pieceType === "Pawn" && getCol(from) !== getCol(to)) {
    const capturedPawnSquare = getSquareFromRowCol(getRow(from), getCol(to));
    const enPassantPiece = pieces.find((p) => p.square === capturedPawnSquare);
    if (enPassantPiece) {
      capturedPieceType = enPassantPiece.pieceType;
      await db.delete(schema.pieces).where(and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, capturedPawnSquare)));
    }
  }

  if (movingPiece.pieceType === "King" && Math.abs(getCol(to) - getCol(from)) === 2) {
    const isKingside = getCol(to) === 6;
    const rookFromSquare = getSquareFromRowCol(getRow(from), isKingside ? 7 : 0);
    const rookToSquare = getSquareFromRowCol(getRow(from), isKingside ? 5 : 3);
    await db.update(schema.pieces)
      .set({ square: rookToSquare, hasMoved: true })
      .where(and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, rookFromSquare)));
  }

  let finalPieceType = movingPiece.pieceType;
  if (movingPiece.pieceType === "Pawn") {
    const targetRow = movingPiece.color === "White" ? 0 : 7;
    if (getRow(to) === targetRow) finalPieceType = "Queen";
  }

  await db.update(schema.pieces)
    .set({ square: to, hasMoved: true, pieceType: finalPieceType })
    .where(and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, from)));

  const moveCount = await db.query.moves.findMany({ where: eq(schema.moves.gameId, currentGame.id) });
  await db.insert(schema.moves).values({
    gameId: currentGame.id,
    fromSquare: from,
    toSquare: to,
    pieceType: movingPiece.pieceType,
    pieceColor: movingPiece.color,
    capturedPieceType,
    moveNumber: moveCount.length + 1,
    createdAt: now,
  });

  const updatedPieces = await db.query.pieces.findMany({ where: eq(schema.pieces.gameId, currentGame.id) });
  const nextTurn: Color = currentGame.currentTurn === "White" ? "Black" : "White";
  const currentMove = await db.query.moves.findFirst({
    where: eq(schema.moves.gameId, currentGame.id),
    orderBy: desc(schema.moves.moveNumber),
  });
  const newStatus = getGameStatus(updatedPieces, nextTurn, currentMove ?? undefined);

  await db.update(schema.games)
    .set({ currentTurn: nextTurn, status: newStatus, updatedAt: now, lastMoveTime: now })
    .where(eq(schema.games.id, currentGame.id));

  if (nextTurn === "Black" && newStatus === "Ongoing") {
    const fen = piecesToFen(updatedPieces, nextTurn, currentMove ?? undefined);
    const engineUrl = process.env.CHESS_ENGINE_URL || "http://127.0.0.1:8080";
    fetch(`${engineUrl}/api/engine-move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen, engine_type: engineType }),
    })
      .then(async (response) => {
        if (!response.ok) return;
        const data = await response.json() as { best_move?: string };
        if (data.best_move) {
          const uciMove = data.best_move;
          const fromFile = uciMove.charCodeAt(0) - 97;
          const fromRank = 8 - parseInt(uciMove[1]);
          const toFile = uciMove.charCodeAt(2) - 97;
          const toRank = 8 - parseInt(uciMove[3]);
          await applyEngineMove(currentGame.id, getSquareFromRowCol(fromRank, fromFile), getSquareFromRowCol(toRank, toFile));
        }
      })
      .catch((e) => console.error("Engine move failed:", e));
  }

  return c.json({ success: true, nextTurn, status: newStatus });
});

app.post("/api/undo-move", async (c) => {
  const { gameId } = await c.req.json<{ gameId: number }>();
  const currentGame = await db.query.games.findFirst({ where: eq(schema.games.id, gameId) });
  if (!currentGame) return c.json({ error: "No game found" }, 404);

  const lastMove = await db.query.moves.findFirst({
    where: eq(schema.moves.gameId, currentGame.id),
    orderBy: desc(schema.moves.moveNumber),
  });
  if (!lastMove) return c.json({ success: false, message: "No moves to undo" });

  const movedPiece = await db.query.pieces.findFirst({
    where: and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, lastMove.toSquare)),
  });

  if (movedPiece) {
    await db.update(schema.pieces)
      .set({ square: lastMove.fromSquare, pieceType: lastMove.pieceType, hasMoved: false })
      .where(eq(schema.pieces.id, movedPiece.id));
  }

  if (lastMove.capturedPieceType) {
    const capturedColor = lastMove.pieceColor === "White" ? "Black" : "White";
    await db.insert(schema.pieces).values({
      gameId: currentGame.id,
      color: capturedColor,
      pieceType: lastMove.capturedPieceType,
      square: lastMove.toSquare,
      hasMoved: true,
    });
  }

  if (lastMove.pieceType === "King" && Math.abs(lastMove.fromSquare - lastMove.toSquare) === 2) {
    const isKingside = getCol(lastMove.toSquare) === 6;
    const row = getRow(lastMove.fromSquare);
    const rookLandedSquare = getSquareFromRowCol(row, isKingside ? 5 : 3);
    const rookOriginalSquare = getSquareFromRowCol(row, isKingside ? 7 : 0);
    await db.update(schema.pieces)
      .set({ square: rookOriginalSquare, hasMoved: false })
      .where(and(eq(schema.pieces.gameId, currentGame.id), eq(schema.pieces.square, rookLandedSquare), eq(schema.pieces.pieceType, "Rook")));
  }

  await db.delete(schema.moves).where(eq(schema.moves.id, lastMove.id));
  await db.update(schema.games)
    .set({ currentTurn: lastMove.pieceColor, status: "Ongoing", updatedAt: Date.now() })
    .where(eq(schema.games.id, currentGame.id));

  return c.json({ success: true });
});

app.post("/api/reset", async (c) => {
  const gamesToDelete = await db.query.games.findMany({ where: eq(schema.games.mode, "vs_computer") });
  const gameIds = gamesToDelete.map((g) => g.id);
  if (gameIds.length > 0) {
    await db.delete(schema.moves).where(inArray(schema.moves.gameId, gameIds));
    await db.delete(schema.pieces).where(inArray(schema.pieces.gameId, gameIds));
    await db.delete(schema.games).where(inArray(schema.games.id, gameIds));
  }
  const initialData = initializeGame();
  const [newGame] = await db
    .insert(schema.games)
    .values({
      currentTurn: initialData.turn,
      status: "Ongoing",
      mode: "vs_computer",
      timeControl: 0,
      whiteTimeRemaining: Number.MAX_SAFE_INTEGER,
      blackTimeRemaining: Number.MAX_SAFE_INTEGER,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastMoveTime: null,
    })
    .returning();
  if (newGame) {
    await db.insert(schema.pieces).values(
      initialData.pieces.map((p) => ({
        gameId: newGame.id,
        color: p.color,
        pieceType: p.pieceType,
        square: p.square,
        hasMoved: false,
      })),
    );
  }
  return c.json({ success: true });
});

// ─── Static files + SPA fallback ─────────────────────────────────────────────

app.use("/*", serveStatic({ root: "./dist" }));
app.get("/*", async (c) => {
  const html = await Bun.file("./dist/index.html").text();
  return c.html(html);
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function applyEngineMove(gameId: number, from: number, to: number) {
  const pieces = await db.query.pieces.findMany({ where: eq(schema.pieces.gameId, gameId) });
  const movingPiece = pieces.find((p) => p.square === from);
  if (!movingPiece) return;

  const capturedPiece = pieces.find((p) => p.square === to);
  let capturedPieceType: PieceType | undefined;

  if (capturedPiece) {
    capturedPieceType = capturedPiece.pieceType;
    await db.delete(schema.pieces).where(and(eq(schema.pieces.gameId, gameId), eq(schema.pieces.square, to)));
  } else if (movingPiece.pieceType === "Pawn" && getCol(from) !== getCol(to)) {
    const capturedPawnSquare = getSquareFromRowCol(getRow(from), getCol(to));
    const enPassantPiece = pieces.find((p) => p.square === capturedPawnSquare);
    if (enPassantPiece) {
      capturedPieceType = enPassantPiece.pieceType;
      await db.delete(schema.pieces).where(and(eq(schema.pieces.gameId, gameId), eq(schema.pieces.square, capturedPawnSquare)));
    }
  }

  if (movingPiece.pieceType === "King" && Math.abs(getCol(to) - getCol(from)) === 2) {
    const isKingside = getCol(to) === 6;
    const rookFromSquare = getSquareFromRowCol(getRow(from), isKingside ? 7 : 0);
    const rookToSquare = getSquareFromRowCol(getRow(from), isKingside ? 5 : 3);
    await db.update(schema.pieces)
      .set({ square: rookToSquare, hasMoved: true })
      .where(and(eq(schema.pieces.gameId, gameId), eq(schema.pieces.square, rookFromSquare)));
  }

  let finalPieceType = movingPiece.pieceType;
  if (movingPiece.pieceType === "Pawn") {
    const targetRow = movingPiece.color === "White" ? 0 : 7;
    if (getRow(to) === targetRow) finalPieceType = "Queen";
  }

  await db.update(schema.pieces)
    .set({ square: to, hasMoved: true, pieceType: finalPieceType })
    .where(and(eq(schema.pieces.gameId, gameId), eq(schema.pieces.square, from)));

  const now = Date.now();
  const moveCount = await db.query.moves.findMany({ where: eq(schema.moves.gameId, gameId) });
  await db.insert(schema.moves).values({
    gameId,
    fromSquare: from,
    toSquare: to,
    pieceType: movingPiece.pieceType,
    pieceColor: movingPiece.color,
    capturedPieceType,
    moveNumber: moveCount.length + 1,
    createdAt: now,
  });

  const updatedPieces = await db.query.pieces.findMany({ where: eq(schema.pieces.gameId, gameId) });
  const nextTurn: Color = movingPiece.color === "White" ? "Black" : "White";
  const lastMove = await db.query.moves.findFirst({
    where: eq(schema.moves.gameId, gameId),
    orderBy: desc(schema.moves.moveNumber),
  });
  const newStatus = getGameStatus(updatedPieces, nextTurn, lastMove ?? undefined);

  await db.update(schema.games)
    .set({ currentTurn: nextTurn, status: newStatus, updatedAt: now, lastMoveTime: now })
    .where(eq(schema.games.id, gameId));
}

export default { port: 3000, fetch: app.fetch };
