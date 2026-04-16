use shakmaty::{Chess, Color, Move, Position};

use crate::eval::evaluate;

fn minimax(pos: &Chess, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
    if depth == 0 || pos.is_game_over() {
        return evaluate(pos);
    }
    let legals = pos.legal_moves();
    if maximizing {
        let mut max_eval = i32::MIN;
        for m in legals {
            let mut np = pos.clone();
            np.play_unchecked(&m);
            let e = minimax(&np, depth - 1, alpha, beta, false);
            max_eval = max_eval.max(e);
            alpha = alpha.max(e);
            if beta <= alpha { break; }
        }
        max_eval
    } else {
        let mut min_eval = i32::MAX;
        for m in legals {
            let mut np = pos.clone();
            np.play_unchecked(&m);
            let e = minimax(&np, depth - 1, alpha, beta, true);
            min_eval = min_eval.min(e);
            beta = beta.min(e);
            if beta <= alpha { break; }
        }
        min_eval
    }
}

pub fn minimax_best_move(pos: &Chess, depth: i32) -> Option<Move> {
    let legals = pos.legal_moves();
    if legals.is_empty() { return None; }
    let maximizing = pos.turn() == Color::White;
    let mut best_move = None;
    if maximizing {
        let mut max_eval = i32::MIN;
        for m in legals {
            let mut np = pos.clone();
            np.play_unchecked(&m);
            let e = minimax(&np, depth - 1, i32::MIN, i32::MAX, false);
            if e > max_eval { max_eval = e; best_move = Some(m); }
        }
    } else {
        let mut min_eval = i32::MAX;
        for m in legals {
            let mut np = pos.clone();
            np.play_unchecked(&m);
            let e = minimax(&np, depth - 1, i32::MIN, i32::MAX, true);
            if e < min_eval { min_eval = e; best_move = Some(m); }
        }
    }
    best_move
}
