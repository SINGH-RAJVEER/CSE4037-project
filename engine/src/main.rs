mod board;
mod eval;
mod handlers;
mod mcts;
mod minimax;
mod model;

use actix_cors::Cors;
use actix_web::{App, HttpServer};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    model::load_model();
    println!("Chess engine at http://0.0.0.0:8080");
    println!("POST /api/engine-move  {{ fen, engine_type: \"minimax\" | \"neural\" }}");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
        App::new()
            .wrap(cors)
            .service(handlers::health)
            .service(handlers::engine_move)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
