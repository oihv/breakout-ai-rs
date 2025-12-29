mod neat;
mod config;
use macroquad::prelude::*;

#[macroquad::main("Arkanoid")]
async fn main() {
    const BLOCKS_W: usize = 10;
    const BLOCKS_H: usize = 10;
    const SCR_W: f32 = 20.0;
    const SCR_H: f32 = 20.0;

    let mut blocks: [[bool; BLOCKS_W]; BLOCKS_H] = [[true; BLOCKS_W]; BLOCKS_H];
    let mut ball_x = 12.;
    let mut ball_y = 7.;
    let mut dx = 6.5;
    let mut dy = -6.5;
    let mut platform_x = 10.;
    let mut stick = true;
    let platform_width = 5.;
    let platform_height = 0.2;
    let mut score = 0;
    let mut elapsed_time = 0.;
    let player_speed = 8.0;
    let global_speed = 0.5;

    // build camera with following coordinate system:
    // (0., 0)     .... (SCR_W, 0.)
    // (0., SCR_H) .... (SCR_W, SCR_H)
    set_camera(&Camera2D {
        zoom: vec2(1. / SCR_W * 2., 1. / SCR_H * 2.),
        target: vec2(SCR_W / 2., SCR_H / 2.),
        ..Default::default()
    });

    loop {
        clear_background(SKYBLUE);

        let delta = get_frame_time();

        if is_key_down(KeyCode::Right) && platform_x < SCR_W - platform_width / 2. {
            platform_x += player_speed * delta * global_speed;
        }
        if is_key_down(KeyCode::Left) && platform_x > platform_width / 2. {
            platform_x -= player_speed * delta * global_speed;
        }

        if stick == false {
            ball_x += dx * delta * global_speed;
            ball_y += dy * delta * global_speed;
            elapsed_time += delta * global_speed;
        } else {
            let (font_size, font_scale, font_aspect) = camera_font_scale(1.);
            let text_params = TextParams {
                font_size,
                font_scale,
                font_scale_aspect: font_aspect,
                ..Default::default()
            };
            draw_text_ex(
                "Press space to start",
                SCR_W / 2. - 3.5,
                2.,
                text_params,
            );

            ball_x = platform_x;
            ball_y = SCR_H - 0.5;

            stick = !is_key_down(KeyCode::Space);
        }

        if ball_x <= 0. || ball_x > SCR_W {
            dx *= -1.;
        }
        if ball_y <= 0.
            || (ball_y > SCR_H - platform_height - 0.15 / 2.
                && ball_x >= platform_x - platform_width / 2.
                && ball_x <= platform_x + platform_width / 2.)
        {
            dy *= -1.;
        }
        if ball_y >= SCR_H {
            ball_y = 10.;
            dy = -dy.abs();
            stick = true;
        }

        for j in 0..BLOCKS_H {
            for i in 0..BLOCKS_W {
                if blocks[j][i] {
                    let block_w = SCR_W / BLOCKS_W as f32;
                    let block_h = 7.0 / BLOCKS_H as f32;
                    let block_x = i as f32 * block_w + 0.05;
                    let block_y = j as f32 * block_h + 0.05 + 3.;

                    draw_rectangle(block_x, block_y, block_w - 0.1, block_h - 0.1, DARKBLUE);
                    if ball_x >= block_x
                        && ball_x < block_x + block_w
                        && ball_y >= block_y
                        && ball_y < block_y + block_h
                    {
                        dy *= -1.;
                        blocks[j][i] = false;
                        score += 10;
                    }
                }
            }
        }

        let (font_size, font_scale, font_aspect) = camera_font_scale(1.);
        let text_params = TextParams {
            font_size,
            font_scale,
            font_scale_aspect: font_aspect,
            ..Default::default()
        };
        let score_text = format!("Score: {score}");
        draw_text_ex(
            &score_text,
            0.5,
            1.,
            text_params.clone(),
        );
        let score_text = format!("elapsed time: {elapsed_time}");
        draw_text_ex(
            &score_text,
            4.5,
            1.,
            text_params,
        );
        draw_circle(ball_x, ball_y, 0.2, RED);
        draw_rectangle(
            platform_x - platform_width / 2.,
            SCR_H - platform_height,
            platform_width,
            platform_height,
            DARKPURPLE,
        );

        next_frame().await
    }
}
