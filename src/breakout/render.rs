use macroquad::prelude::*;
use super::engine::BreakoutEngine;

pub fn render_game(engine: &BreakoutEngine) {
    clear_background(SKYBLUE);
    
    // Draw blocks
    for j in 0..engine.blocks_h {
        for i in 0..engine.blocks_w {
            if engine.blocks[j][i] {
                let block_w = engine.scr_w / engine.blocks_w as f32;
                let block_h = 7.0 / engine.blocks_h as f32;
                let block_x = i as f32 * block_w + 0.05;
                let block_y = j as f32 * block_h + 0.05 + 3.0;
                
                draw_rectangle(block_x, block_y, block_w - 0.1, block_h - 0.1, DARKBLUE);
            }
        }
    }
    
    // Draw score and time
    let (font_size, font_scale, font_aspect) = camera_font_scale(1.0);
    let text_params = TextParams {
        font_size,
        font_scale,
        font_scale_aspect: font_aspect,
        ..Default::default()
    };
    
    let score_text = format!("Score: {}", engine.score);
    draw_text_ex(&score_text, 0.5, 1.0, text_params.clone());
    
    let time_text = format!("Elapsed time: {:.1}", engine.elapsed_time);
    draw_text_ex(&time_text, 4.5, 1.0, text_params.clone());
    
    // Draw "press space to start" message if on stick
    if engine.stick {
        draw_text_ex(
            "Press space to start",
            engine.scr_w / 2.0 - 3.5,
            2.0,
            text_params.clone(),
        );
    }
    
    // Draw ball
    draw_circle(engine.ball_x, engine.ball_y, 0.2, RED);
    
    // Draw platform
    draw_rectangle(
        engine.platform_x - engine.platform_width / 2.0,
        engine.scr_h - engine.platform_height,
        engine.platform_width,
        engine.platform_height,
        DARKPURPLE,
    );
    
    // Draw game over message
    if engine.game_over {
        draw_text_ex(
            "Game Over!",
            engine.scr_w / 2.0 - 2.0,
            engine.scr_h / 2.0,
            text_params.clone(),
        );
    }
}

pub fn setup_camera(scr_w: f32, scr_h: f32) {
    set_camera(&Camera2D {
        zoom: vec2(1.0 / scr_w * 2.0, 1.0 / scr_h * 2.0),
        target: vec2(scr_w / 2.0, scr_h / 2.0),
        ..Default::default()
    });
}
