use actix_web::{get, post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::{CastlingMode, Chess};

use crate::mcts::neural_best_move;
use crate::minimax::minimax_best_move;
use crate::model::NN_SESSION;

#[derive(Deserialize)]
pub struct EngineRequest {
    pub fen: String,
    /// "minimax" | "neural"
    pub engine_type: Option<String>,
}

#[derive(Serialize)]
pub struct EngineResponse {
    pub best_move: Option<String>,
    pub error: Option<String>,
}

#[get("/health")]
pub async fn health() -> impl Responder {
    HttpResponse::Ok().finish()
}

#[post("/api/engine-move")]
pub async fn engine_move(req: web::Json<EngineRequest>) -> impl Responder {
    let setup: Fen = match req.fen.parse() {
        Ok(f)  => f,
        Err(_) => return HttpResponse::BadRequest().json(EngineResponse {
            best_move: None, error: Some("Invalid FEN".into()),
        }),
    };
    let position: Chess = match setup.into_position(CastlingMode::Standard) {
        Ok(p)  => p,
        Err(_) => return HttpResponse::BadRequest().json(EngineResponse {
            best_move: None, error: Some("Invalid position".into()),
        }),
    };

    let best_move = match req.engine_type.as_deref().unwrap_or("minimax") {
        "neural" => match NN_SESSION.get() {
            Some(mutex) => neural_best_move(&position, &mut mutex.lock().unwrap()),
            None => {
                eprintln!("[engine] neural requested but no model loaded; falling back to minimax");
                minimax_best_move(&position, 5)
            }
        },
        _ => minimax_best_move(&position, 5),
    };

    match best_move {
        Some(m) => {
            let uci = Uci::from_move(&m, CastlingMode::Standard);
            HttpResponse::Ok().json(EngineResponse { best_move: Some(uci.to_string()), error: None })
        }
        None => HttpResponse::Ok().json(EngineResponse {
            best_move: None, error: Some("No legal moves".into()),
        }),
    }
}
