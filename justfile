alias t := train
alias tp := train-prev
alias g := game
train:
  cargo r --bin breakout-train

ai:
  cargo r --bin breakout-ai

aic:
  cargo r --bin breakout-ai -- --champion

train-prev:
  cargo r --bin breakout-train -- --prev

game:
  cargo r --bin breakout-game 
