import type { Color, GameStatus, PieceType } from "../db/schema";

export type BoardResponse = {
  id: number;
  pieces: { color: Color; piece_type: PieceType; square: number }[];
  capturedPieces: { white: PieceType[]; black: PieceType[] };
  moves: {
    from: number;
    to: number;
    color: Color;
    pieceType: PieceType;
    captured?: PieceType;
    notation: string;
  }[];
  turn: Color;
  status: GameStatus;
  lastMove: { from: number; to: number } | null;
  serverTime: number;
};

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText })) as { error?: string };
    throw new Error(err.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const getBoard = () => post<BoardResponse>("/api/board");

export const getMoves = (square: number, gameId: number) =>
  post<number[]>("/api/moves", { square, gameId });

export const makeMove = (from: number, to: number, gameId: number, engineType: "minimax" | "neural") =>
  post<{ success: boolean; nextTurn: Color; status: GameStatus }>("/api/make-move", { from, to, gameId, engineType });

export const undoMove = (gameId: number) =>
  post<{ success: boolean }>("/api/undo-move", { gameId });

export const resetGame = () => post<{ success: boolean }>("/api/reset");

export type { Color, PieceType, GameStatus };
