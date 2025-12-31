mod neat;
mod config;
mod breakout;
mod training;
mod serialization;

use macroquad::prelude::*;
use breakout::{BreakoutEngine, engine::Action, render::{render_game, setup_camera}};

#[macroquad::main("Arkanoid")]
async fn main() {
    let mut engine = BreakoutEngine::new();
    setup_camera(engine.scr_w, engine.scr_h);

    loop {
        let delta = get_frame_time();
        
        // Handle input
        let action = if engine.stick {
            if is_key_down(KeyCode::Space) {
                Action::Start
            } else {
                Action::Stay
            }
        } else if is_key_down(KeyCode::Right) {
            Action::Right
        } else if is_key_down(KeyCode::Left) {
            Action::Left
        } else {
            Action::Stay
        };
        
        // Update game state
        engine.step(action, delta);
        
        // Render
        render_game(&engine);
        
        // Reset on game over
        if engine.game_over && is_key_pressed(KeyCode::R) {
            engine.reset();
        }

        next_frame().await
    }
}
