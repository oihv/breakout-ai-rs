use std::f32::consts::PI;

use rand::rand_core::block;

/// Breakout game engine that can be used for both human play and AI training
#[derive(Clone)]
pub struct BreakoutEngine {
    // Game state
    pub blocks: Vec<Vec<bool>>,
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_rad: f32,
    pub ball_speed: f32,
    pub ball_min_shoot_angle: f32,
    pub dx: f32,
    pub dy: f32,
    pub platform_x: f32,
    pub stick: bool,
    pub score: i32,
    pub elapsed_time: f32,

    // Game constants
    pub blocks_w: usize,
    pub blocks_h: usize,
    pub scr_w: f32,
    pub scr_h: f32,
    pub platform_width: f32,
    pub platform_height: f32,
    pub player_speed: f32,
    pub global_speed: f32,

    // Game status
    pub game_over: bool,
    pub frames_alive: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Left,
    Right,
    Stay,
    Start,
}

impl BreakoutEngine {
    pub fn new() -> Self {
        const BLOCKS_W: usize = 10;
        const BLOCKS_H: usize = 10;
        const SCR_W: f32 = 20.0;
        const SCR_H: f32 = 20.0;

        let mut blocks = Vec::new();
        for _ in 0..BLOCKS_H {
            let row = vec![true; BLOCKS_W];
            blocks.push(row);
        }

        Self {
            blocks,
            ball_x: 12.0,
            ball_y: 12.0,
            ball_rad: 0.15,
            ball_speed: 10.,
            ball_min_shoot_angle: 30.,
            dx: 6.5,
            dy: -6.5,
            platform_x: 10.0,
            stick: true,
            score: 0,
            elapsed_time: 0.0,
            blocks_w: BLOCKS_W,
            blocks_h: BLOCKS_H,
            scr_w: SCR_W,
            scr_h: SCR_H,
            platform_width: 5.0,
            platform_height: 0.2,
            player_speed: 8.0,
            global_speed: 1.,
            game_over: false,
            frames_alive: 0,
        }
    }

    /// Reset the game to initial state
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get the current game state as a vector of inputs for the neural network
    /// Returns: [ball_x, ball_y, ball_dx, ball_dy, platform_x, blocks_remaining, score, elapsed_time]
    pub fn get_state(&self) -> Vec<f32> {
        let blocks_remaining = self
            .blocks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&b| b)
            .count() as f32;

        vec![
            self.ball_x / self.scr_w, // Normalized ball x position
            self.ball_y / self.scr_h, // Normalized ball y position
            // self.dx / 10.0,                      // Normalized ball x velocity
            // self.dy / 10.0,                      // Normalized ball y velocity
            self.platform_x / self.scr_w, // Normalized platform x position
                                          // blocks_remaining / (self.blocks_w * self.blocks_h) as f32, // Normalized blocks remaining
                                          // self.score as f32 / 1000.0,         // Normalized score
                                          // self.elapsed_time / 100.0,          // Normalized elapsed time
        ]
    }

    // the bouncing of the ball should match the position where the ball collides
    pub fn bounce_ball(&mut self) {
        // 180   90    0
        // xxxxxxxxxxxxx (platform)
        let scale = (self.platform_width
            - (self.ball_x - (self.platform_x - self.platform_width / 2.)))
            / self.platform_width;
        let mut angle_deg = scale * 180.;
        // Normalized = old val * (new_max -  new_min) / (old_max - old_min) + new_min
        angle_deg = angle_deg * (180. - self.ball_min_shoot_angle - self.ball_min_shoot_angle) / (180. - 0.) + self.ball_min_shoot_angle;
        let angle_rad = PI * angle_deg / (180.);

        self.dx = self.ball_speed * f32::cos(angle_rad);
        self.dy = self.ball_speed * f32::sin(angle_rad);
        self.dy *= -1.0;
        self.resolve_collision();
    }

    pub fn resolve_collision(&mut self) {
        self.ball_y = self.scr_h - self.platform_height - self.ball_rad;
    }

    /// Step the game forward by one frame with the given action
    pub fn step(&mut self, action: Action, delta: f32) {
        if self.game_over {
            return;
        }

        self.frames_alive += 1;

        // Handle player/AI action
        match action {
            Action::Left => {
                if self.platform_x > self.platform_width / 2.0 {
                    self.platform_x -= self.player_speed * delta * self.global_speed;
                } else {
                    self.platform_x = self.platform_width / 2.;
                }
            }
            Action::Right => {
                if self.platform_x < self.scr_w - self.platform_width / 2.0 {
                    self.platform_x += self.player_speed * delta * self.global_speed;
                } else {
                    self.platform_x = self.scr_w - self.platform_width / 2.;
                }
            }
            Action::Start => {
                self.stick = false;
            }
            Action::Stay => {}
        }

        // Update ball position
        if !self.stick {
            self.ball_x += self.dx * delta * self.global_speed;
            self.ball_y += self.dy * delta * self.global_speed;
            self.elapsed_time += delta * self.global_speed;
        } else {
            self.ball_x = self.platform_x;
            self.ball_y = self.scr_h - 0.5;
        }

        // Ball collision with walls
        if self.ball_x <= 0.0 || self.ball_x > self.scr_w {
            self.dx *= -1.0;
        }

        // Ball collision platform
        if self.ball_y > self.scr_h - self.platform_height - self.ball_rad / 2.0
            && self.ball_x >= self.platform_x - self.platform_width / 2.0
            && self.ball_x <= self.platform_x + self.platform_width / 2.0
        {
            self.bounce_ball();
        }

        if self.ball_y <= 0.0 {
            self.dy *= -1.;
        }

        // Ball fell through bottom
        if self.ball_y >= self.scr_h {
            self.game_over = true;
            return;
        }

        // Check block collisions
        for j in 0..self.blocks_h {
            for i in 0..self.blocks_w {
                if self.blocks[j][i] {
                    let block_w = self.scr_w / self.blocks_w as f32;
                    let block_h = 7.0 / self.blocks_h as f32;
                    let block_x = i as f32 * block_w + 0.05;
                    let block_y = j as f32 * block_h + 0.05 + 3.0;

                    if self.ball_x >= block_x
                        && self.ball_x < block_x + block_w
                        && self.ball_y >= block_y
                        && self.ball_y < block_y + block_h
                    {
                        self.dy *= -1.0;
                        self.blocks[j][i] = false;
                        self.score += 10;
                    }
                }
            }
        }

        // Check if all blocks are destroyed (win condition)
        let blocks_remaining = self
            .blocks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&b| b)
            .count();

        if blocks_remaining == 0 {
            self.game_over = true;
        }
    }

    /// Calculate fitness for NEAT training
    /// Fitness is based on: time survived (primary), blocks destroyed, and score
    pub fn calculate_fitness(&self) -> f32 {
        let blocks_destroyed = (self.blocks_w * self.blocks_h) as f32
            - self
                .blocks
                .iter()
                .flat_map(|row| row.iter())
                .filter(|&&b| b)
                .count() as f32;

        // Primary reward: surviving longer (each frame is worth 1 point)
        let time_bonus = self.frames_alive as f32 * 1.0;

        // Secondary rewards: destroying blocks and getting score
        let block_bonus = blocks_destroyed * 50.0;
        let score_bonus = self.score as f32;

        // Survival is the most important for initial learning
        time_bonus + block_bonus + score_bonus
    }
}

impl Default for BreakoutEngine {
    fn default() -> Self {
        Self::new()
    }
}
