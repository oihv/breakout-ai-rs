mod breakout;
mod config;
mod neat;
mod serialization;
mod training;

use breakout::{
    BreakoutEngine,
    engine::Action,
    render::{render_game, setup_camera},
};
use macroquad::{prelude::*, text};
use neat::Genome;
use neat::nn::FeedForwardNeuralNetwork;

#[macroquad::main("Arkanoid - AI Playing")]
async fn main() {
    // Load the best genome
    let mut genome: Genome;
    if std::env::args().any(|x| &x == "--champion") {
        genome = match serialization::load_genome("best_of_the_best.pb") {
            Ok(g) => {
                println!("Successfully loaded **CHAMPION** genome!");
                g
            }
            Err(e) => {
                eprintln!("Failed to load genome: {}", e);
                eprintln!("Please run training first: cargo run --release --bin breakout-train");
                return;
            }
        };
    } else {
        genome = match serialization::load_genome("best_genome.pb") {
            Ok(g) => {
                println!("Successfully loaded best last trained genome!");
                g
            }
            Err(e) => {
                eprintln!("Failed to load genome: {}", e);
                eprintln!("Please run training first: cargo run --release --bin breakout-train");
                return;
            }
        };
    }

    let mut engine = BreakoutEngine::new();
    let mut network = FeedForwardNeuralNetwork::create_from_genome(&genome);

    setup_camera(engine.scr_w, engine.scr_h);

    // Auto-start the game
    engine.stick = false;

    let (font_size, font_scale, font_aspect) = camera_font_scale(1.0);
    let text_params = TextParams {
        font_size,
        font_scale,
        font_scale_aspect: font_aspect,
        color: DARKGRAY,
        ..Default::default()
    };

    loop {
        let delta = get_frame_time();

        if !engine.game_over {
            // Render
            render_game(&engine);

            // Get current game state
            let state = engine.get_state();

            // Run neural network to get action
            let outputs = network.activate(state.clone());

            // Increase global_speed
            if is_key_released(KeyCode::Up) {
                engine.global_speed += 0.1;
            } else if is_key_released(KeyCode::Down) {
                engine.global_speed -= 0.1;
            }

            // Determine action from network outputs
            // outputs[0] = left, outputs[1] = stay, outputs[2] = right
            let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
                Action::Left
            } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
                Action::Right
            } else {
                Action::Stay
            };

            // Display debug info
            draw_text_ex(
                &format!(
                    "Ball: ({:.1}, {:.1})",
                    state[0] * engine.scr_w,
                    state[1] * engine.scr_h
                ),
                0.5,
                13.0,
                text_params.clone(),
            );
            // draw_text(
            //     &format!("Paddle: {:.1}", state[2] * engine.scr_w),
            //     10.0,
            //     120.0,
            //     16.0,
            //     LIGHTGRAY,
            // );
            draw_text_ex(
                &format!(
                    "NN Out: L:{:.2} S:{:.2} R:{:.2}",
                    outputs[0], outputs[1], outputs[2]
                ),
                0.5,
                14.,
                text_params.clone(),
            );
            draw_text_ex(
                &format!("Action: {:#?}", action),
                0.5,
                15.,
                text_params.clone(),
            );
            draw_text_ex(
                &format!("global speed: {:.1}", engine.global_speed),
                0.5,
                16.,
                text_params.clone(),
            );

            // Update game state
            engine.step(action, delta);
        }

        // Display AI info
        draw_text_ex("AI PLAYING", 15.0, 1., text_params.clone());

        if engine.game_over {
            draw_text(
                "GAME OVER",
                engine.scr_w / 2.0 - 80.0,
                engine.scr_h / 2.0,
                font_size.into(),
                RED,
            );
            draw_text(
                "Press R to restart",
                engine.scr_w / 2.0 - 100.0,
                engine.scr_h / 2.0 + 40.0,
                20.0,
                WHITE,
            );
            draw_text(
                &format!("Final Score: {}", engine.score),
                engine.scr_w / 2.0 - 80.0,
                engine.scr_h / 2.0 + 70.0,
                20.0,
                WHITE,
            );
            draw_text(
                &format!("Fitness: {:.0}", engine.calculate_fitness()),
                engine.scr_w / 2.0 - 60.0,
                engine.scr_h / 2.0 + 100.0,
                20.0,
                YELLOW,
            );
        }

        // Reset on game over
        if engine.game_over && is_key_pressed(KeyCode::R) {
            engine.reset();
            network = FeedForwardNeuralNetwork::create_from_genome(&genome);
            engine.stick = false;
        }

        next_frame().await
    }
}
