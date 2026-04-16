import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getBoard, getMoves, makeMove, resetGame, undoMove } from "../lib/api";
import type { Color, PieceType } from "../lib/api";

type EngineType = "minimax" | "neural";

const PIECE_SYMBOLS: Record<Color, Record<PieceType, string>> = {
  White: { Pawn: "♙", Knight: "♘", Bishop: "♗", Rook: "♖", Queen: "♕", King: "♔" },
  Black: { Pawn: "♟", Knight: "♞", Bishop: "♝", Rook: "♜", Queen: "♛", King: "♚" },
};

function loadEngineType(): EngineType {
  if (typeof window === "undefined") return "minimax";
  try {
    const stored = localStorage.getItem("chess_engine_type");
    if (stored === "neural" || stored === "minimax") return stored;
  } catch (_) {}
  return "minimax";
}

function loadPendingMove() {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem("chess_pending_move");
    return stored ? (JSON.parse(stored) as { from: number; to: number }) : null;
  } catch (_) {
    return null;
  }
}

const S = {
  root: {
    height: "100dvh",
    display: "flex",
    flexDirection: "column" as const,
    background: "#0c0c0c",
    color: "#c8c4bc",
    fontFamily: "ui-monospace, 'SF Mono', 'Fira Code', monospace",
    overflow: "hidden",
  },
  boardArea: {
    flex: "1",
    minHeight: "0",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "8px 8px 0",
  },
  boardWrapper: (size: string) => ({
    position: "relative" as const,
    width: size,
    height: size,
  }),
  board: {
    width: "100%",
    height: "100%",
    display: "grid",
    gridTemplateColumns: "repeat(8, 1fr)",
    gridTemplateRows: "repeat(8, 1fr)",
    border: "1px solid #2a2a2a",
    position: "relative" as const,
  },
  controls: {
    height: "44px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "0 12px",
    borderTop: "1px solid #1e1e1e",
    flexShrink: "0" as const,
  },
  history: {
    height: "28px",
    overflowX: "auto" as const,
    overflowY: "hidden" as const,
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "0 12px",
    borderTop: "1px solid #161616",
    flexShrink: "0" as const,
    scrollbarWidth: "none" as const,
  },
  btn: (active?: boolean, color?: string) => ({
    padding: "3px 10px",
    background: active ? (color ?? "#1e1b4b") : "transparent",
    border: `1px solid ${active ? (color ? color.replace("1b4b", "3730a3") : "#4338ca") : "#2e2e2e"}`,
    color: active ? (color ? "#a7f3d0" : "#a5b4fc") : "#6b7280",
    fontFamily: "inherit",
    fontSize: "11px",
    fontWeight: "bold",
    textTransform: "uppercase" as const,
    letterSpacing: "0.07em",
    cursor: "pointer",
    transition: "opacity 0.1s",
  }),
  btnGreen: {
    padding: "3px 12px",
    background: "#052e16",
    border: "1px solid #166534",
    color: "#86efac",
    fontFamily: "inherit",
    fontSize: "11px",
    fontWeight: "bold",
    textTransform: "uppercase" as const,
    letterSpacing: "0.07em",
    cursor: "pointer",
  },
};

export default function ChessGame() {
  const queryClient = useQueryClient();

  const [engineType, setEngineTypeState] = useState<EngineType>(loadEngineType);
  const setEngineType = (t: EngineType) => {
    setEngineTypeState(t);
    try { localStorage.setItem("chess_engine_type", t); } catch (_) {}
  };

  const [pendingMove, setPendingMoveState] = useState<{ from: number; to: number } | null>(loadPendingMove);
  const setPendingMove = (m: { from: number; to: number } | null) => {
    setPendingMoveState(m);
    try {
      if (m) localStorage.setItem("chess_pending_move", JSON.stringify(m));
      else localStorage.removeItem("chess_pending_move");
    } catch (_) {}
  };

  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [validMoves, setValidMoves] = useState<number[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const boardQuery = useQuery({
    queryKey: ["board"],
    queryFn: () => getBoard(),
    staleTime: 0,
    refetchOnWindowFocus: true,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && data.turn === "Black" && data.status === "Ongoing") return 1000;
      return false;
    },
  });

  useEffect(() => {
    if (boardQuery.data && (boardQuery.data as { mode?: string }).mode === "vs_player") {
      resetMutation.mutate();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boardQuery.data]);

  const moveMutation = useMutation({
    mutationFn: (args: { from: number; to: number }) => {
      if (!boardQuery.data?.id) throw new Error("Game ID not found");
      return makeMove(args.from, args.to, boardQuery.data.id, engineType);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["board"] });
      setPendingMove(null);
    },
    onError: (e: unknown) => {
      const msg = e && typeof e === "object" && "message" in e ? (e as Error).message : String(e);
      setErrorMsg(`Move failed: ${msg}`);
      setTimeout(() => setErrorMsg(null), 3000);
    },
  });

  const undoMutation = useMutation({
    mutationFn: async () => {
      if (!boardQuery.data?.id) throw new Error("Game ID not found");
      const moveCount = boardQuery.data?.moves?.length ?? 0;
      if (moveCount >= 2) {
        await undoMove(boardQuery.data.id);
        await undoMove(boardQuery.data.id);
      } else if (moveCount === 1) {
        await undoMove(boardQuery.data.id);
      }
    },
    onSuccess: async () => { await queryClient.invalidateQueries({ queryKey: ["board"] }); },
    onError: (e: Error) => {
      setErrorMsg(`Takeback failed: ${e.message}`);
      setTimeout(() => setErrorMsg(null), 3000);
    },
  });

  const resetMutation = useMutation({
    mutationFn: () => resetGame(),
    onSuccess: async () => {
      await boardQuery.refetch();
      setSelectedSquare(null);
      setValidMoves([]);
      setPendingMove(null);
      setErrorMsg(null);
    },
    onError: (e: Error) => {
      setErrorMsg(`Reset failed: ${e.message}`);
    },
  });

  const pieces = useMemo(() => {
    const base = boardQuery.data?.pieces ?? [];
    if (!pendingMove) return base;
    return base
      .filter((p) => p.square !== pendingMove.to)
      .map((p) => (p.square === pendingMove.from ? { ...p, square: pendingMove.to } : p));
  }, [boardQuery.data, pendingMove]);

  const turn = boardQuery.data?.turn ?? "White";
  const getPieceAt = (sq: number) => pieces.find((p) => p.square === sq);

  const handleSquareClick = async (sq: number) => {
    if (!boardQuery.data) return;
    if (boardQuery.data.status !== "Ongoing") return;
    if (turn !== "White") return;
    if (pendingMove) return;

    const clickedPiece = pieces.find((p) => p.square === sq);

    if (clickedPiece && clickedPiece.color === "White") {
      if (selectedSquare === sq) { setSelectedSquare(null); setValidMoves([]); return; }
      setSelectedSquare(sq);
      setErrorMsg(null);
      try {
        const moves = await getMoves(sq, boardQuery.data.id);
        setValidMoves(Array.isArray(moves) ? moves : []);
      } catch (e: unknown) {
        setErrorMsg(`Error: ${e instanceof Error ? e.message : "Unknown"}`);
        setValidMoves([]);
      }
      return;
    }

    if (selectedSquare !== null) {
      if (validMoves.includes(sq)) {
        setPendingMove({ from: selectedSquare, to: sq });
        setSelectedSquare(null);
        setValidMoves([]);
      } else {
        setSelectedSquare(null);
        setValidMoves([]);
      }
    }
  };

  const boardSize = "min(calc(100vw - 16px), calc(100dvh - 80px))";

  return (
    <div style={S.root}>
      <div style={S.boardArea}>
        <div style={S.boardWrapper(boardSize)}>
          {errorMsg && (
            <div style={{
              position: "absolute", top: "-32px", left: "50%", transform: "translateX(-50%)",
              background: "#450a0a", color: "#fca5a5", border: "1px solid #7f1d1d",
              padding: "3px 14px", fontSize: "11px", fontWeight: "bold",
              whiteSpace: "nowrap", zIndex: 50,
            }}>
              {errorMsg}
            </div>
          )}

          <div style={S.board}>
            {Array.from({ length: 64 }, (_, sq) => {
              const row = Math.floor(sq / 8);
              const col = sq % 8;
              const isDark = (row + col) % 2 === 1;
              const lm = boardQuery.data?.lastMove;
              const isLastMove = lm && (lm.from === sq || lm.to === sq);
              const isSelected = selectedSquare === sq;
              const piece = getPieceAt(sq);
              const hasPiece = !!piece;

              return (
                <button
                  key={sq}
                  style={{
                    position: "relative",
                    width: "100%",
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: isDark ? "#4a4542" : "#c8c4b8",
                    border: "none",
                    padding: "0",
                    cursor: "pointer",
                    boxShadow: isSelected ? "inset 0 0 0 3px #6366f1" : "none",
                    zIndex: isSelected ? 10 : "auto",
                  }}
                  onClick={() => handleSquareClick(sq)}
                >
                  {isLastMove && (
                    <div style={{ position: "absolute", inset: "0", background: "rgba(253,224,71,0.3)", pointerEvents: "none" }} />
                  )}
                  {col === 0 && (
                    <span style={{
                      position: "absolute", left: "2px", top: "1px",
                      fontSize: "clamp(7px, 1.2vmin, 10px)", fontWeight: "bold",
                      opacity: "0.5", color: isDark ? "#c8c4b8" : "#4a4542",
                      pointerEvents: "none", lineHeight: "1",
                    }}>{8 - row}</span>
                  )}
                  {row === 7 && (
                    <span style={{
                      position: "absolute", right: "2px", bottom: "1px",
                      fontSize: "clamp(7px, 1.2vmin, 10px)", fontWeight: "bold",
                      opacity: "0.5", color: isDark ? "#c8c4b8" : "#4a4542",
                      pointerEvents: "none", lineHeight: "1",
                    }}>{String.fromCharCode(97 + col)}</span>
                  )}
                  {validMoves.includes(sq) && (
                    hasPiece
                      ? <div style={{ position: "absolute", inset: "0", border: "3px solid rgba(220,38,38,0.55)", pointerEvents: "none" }} />
                      : <div style={{ width: "28%", height: "28%", background: "rgba(0,0,0,0.22)", borderRadius: "50%", pointerEvents: "none" }} />
                  )}
                  {piece && (
                    <span style={{
                      fontSize: "clamp(16px, 5.5vmin, 64px)",
                      color: piece.color === "White" ? "#f0ede6" : "#181818",
                      textShadow: piece.color === "White"
                        ? "0 1px 4px rgba(0,0,0,0.55)"
                        : "0 1px 4px rgba(255,255,255,0.18)",
                      userSelect: "none",
                      zIndex: 20,
                      position: "relative",
                    }}>
                      {PIECE_SYMBOLS[piece.color][piece.piece_type]}
                    </span>
                  )}
                </button>
              );
            })}

            {boardQuery.data?.status && boardQuery.data.status !== "Ongoing" && (
              <div style={{
                position: "absolute", inset: "0",
                background: "rgba(0,0,0,0.72)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 50, backdropFilter: "blur(3px)",
              }}>
                <div style={{
                  textAlign: "center", padding: "32px 48px",
                  border: "1px solid #2a2a2a", background: "#0f0f0f",
                }}>
                  <div style={{
                    fontSize: "clamp(22px, 5vmin, 52px)",
                    fontWeight: "900", letterSpacing: "-0.02em", color: "#f0ede6",
                    marginBottom: "4px",
                  }}>
                    {boardQuery.data.status}
                  </div>
                  <div style={{
                    fontSize: "clamp(11px, 1.8vmin, 16px)", color: "#6366f1",
                    textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: "28px",
                  }}>
                    {boardQuery.data.status === "Stalemate"
                      ? "Draw"
                      : turn === "White" ? "Bot wins" : "You win"}
                  </div>
                  <button
                    onClick={() => resetMutation.mutate()}
                    style={{
                      padding: "8px 28px", background: "#312e81", border: "1px solid #4338ca",
                      color: "#c7d2fe", fontFamily: "inherit", fontSize: "12px",
                      fontWeight: "bold", textTransform: "uppercase",
                      letterSpacing: "0.08em", cursor: "pointer",
                    }}
                  >
                    Play Again
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={S.controls}>
        <button
          onClick={() => setEngineType(engineType === "minimax" ? "neural" : "minimax")}
          style={S.btn(true, engineType === "neural" ? "#052e16" : undefined)}
        >
          {engineType === "minimax" ? "Minimax" : "Neural"}
        </button>

        <div style={{ width: "1px", height: "20px", background: "#2a2a2a" }} />

        <div style={{
          flex: "1", textAlign: "center", fontSize: "11px",
          textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: "bold",
          color: turn === "Black" ? "#4b5563" : "#9ca3af",
        }}>
          {boardQuery.data?.status !== "Ongoing"
            ? (boardQuery.data?.status ?? "")
            : turn === "Black" ? "thinking..." : "your move"}
        </div>

        <div style={{ width: "1px", height: "20px", background: "#2a2a2a" }} />

        {pendingMove && turn === "White" && (
          <>
            <button
              onClick={() => moveMutation.mutate(pendingMove)}
              disabled={moveMutation.isPending}
              style={S.btnGreen}
            >
              {moveMutation.isPending ? "..." : "Confirm"}
            </button>
            <button
              onClick={() => setPendingMove(null)}
              disabled={moveMutation.isPending}
              style={S.btn()}
            >
              Cancel
            </button>
          </>
        )}

        {!pendingMove && (
          <button
            onClick={() => undoMutation.mutate()}
            disabled={undoMutation.isPending || (boardQuery.data?.moves?.length ?? 0) === 0}
            style={{ ...S.btn(), opacity: (boardQuery.data?.moves?.length ?? 0) === 0 ? "0.25" : "1" }}
          >
            {undoMutation.isPending ? "..." : "Takeback"}
          </button>
        )}

        <button
          onClick={() => resetMutation.mutate()}
          disabled={resetMutation.isPending}
          style={S.btn()}
        >
          {resetMutation.isPending ? "..." : "New Game"}
        </button>
      </div>

      <div style={S.history}>
        {(boardQuery.data?.moves?.length ?? 0) === 0 ? (
          <span style={{ fontSize: "11px", color: "#2a2a2a", fontStyle: "italic" }}>—</span>
        ) : (
          Array.from({ length: Math.ceil((boardQuery.data?.moves?.length ?? 0) / 2) }, (_, i) => {
            const wm = boardQuery.data?.moves?.[i * 2];
            const bm = boardQuery.data?.moves?.[i * 2 + 1];
            return (
              <span key={i} style={{ fontSize: "11px", whiteSpace: "nowrap" }}>
                <span style={{ color: "#374151", marginRight: "3px" }}>{i + 1}.</span>
                <span style={{ color: "#c9c5bd", marginRight: "4px" }}>{wm?.notation}</span>
                {bm && <span style={{ color: "#6b7280" }}>{bm.notation}</span>}
              </span>
            );
          })
        )}
      </div>
    </div>
  );
}
