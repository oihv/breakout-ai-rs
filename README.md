# Breakout AI - NEAT Implementation in Rust

A project for learning Artificial Intelligence from the ground up.

Made with the intention to learn how Artificial Intelligence is really written
under the hood, and written for the course project is only a plus.

## Acknowledgement
This project is heavily inspired by:
- Google DeepMind's Q Learning Atari breakout 
Link: https://www.youtube.com/watch?v=V1eYniJ0Rnk
The expectation of the project is to turn out somewhat similar to the project
here. And as it turns out (spoiler alert!) this project sucessfully mimics the
"optimal" strategy similar to the one in the Google DeepMind's model. Which is
to make a breakout so that the ball can oscillates between the blocks, thus
achieving high score in a short period of time.

- Tech With Nikola's Snake learns with NEUROEVOLUTION (implementing NEAT from scratch in C++) 
Link: https://www.youtube.com/watch?v=lAjcH-hCusg
The project heavily follows from Nikola's beautiful breakdown of the project,
huge appreciation for his effort making this video, without the walkthrough of
all the different concept, the architecture, and how NEAT is implemented, this
project would be impossible to complete.

- NEAT-python
Link: https://github.com/CodeReclaimers/neat-python/
This repo is also what Tech With Nikola references from time to time in this
video, I followed the approach used in this project, such as how to generate the
NN from genome, and other internal capabilities needed for NEAT to work.

## Theory
This project uses NEAT (NeuroEvolution Augmented Topologies)
Original paper: https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf

Basically, it is a genetic algorithm approach for reinforcement learning,
essentially we start up with a generation that have some amount of individuals,
then all these individuals will be evaluated based on a fitness function.

The top n% individuals will then survive to the next generation, and then
population of that new generation will be created based on the *cross-over* and
*mutation* of the top survivors from the previous generation. 

What's being cross-over-ed and mutated are the genotypes of the neural network,
which in this algorithm is described as genomes. Genomes are collections of
link gene (that contains input id, output id, and weight) and neuron gene (the
neuron id, and bias).

- Cross over selects a random link gene/neuron gene from two different genome,
essentially making a mix of them.
- Mutation randomize the value of the weights/biases in a genome

What this project does is basically all of these, and then connecting the
network to a breakout game, where the input is the position of the ball, 
its velocity, and the platform position, the fitness function is the score of the game,
and the output is where the platform will go.

## Result
The model trained on a reasonable time can finish the game, and even correctly
finds the optimal solution to the game, which is making a tunnel and oscillates
the ball to gain high score in a short period of time.

The parameters that can be changed and their effect in this project are:
- number of steps: it affects how many frames the evaluation process will go,
higher means the network will be tested for late games. But at a certain point,
it will only be stagnant, as the higher stages of the game is usually missing
only a few blocks, or the game is already over.

- number of generations: the number of generations that will be generated, for
this project, the fitness function will also be stuck somewhere meaning there's
no more room for improvement

- other statistical parameters: variables like standard deviation, the
probability of the mutations can also be tweaked, but further research is needed
to understand how will these variables affect the performance of the model.

## Further Improvement
Some improvement that can be made for the model:
- [ ] Take the position of the blocks as the input too so that the model can aim,
thus reducing the time it needed to finish the game
- [ ] Perhaps not giving out the internal information as input directly, and only
enables the NN to assess the situation by image (CNN needed to process the image)
- [ ] Fitness function, the current fitness function rewards genome that survives longer, 
but this shouldn't be the case as the model should focus instead on finishing
faster
- [ ] integrate more algorithms that should be tested whether or not it can work
better for this problem, if so, why

Infrastructure wise, here's the improvements that I have:
- [ ] better code structure, combine all of the functionality into one cohesive
  CLI/GUI program that can interact easily.
- [ ] better visualization of the training process, maybe visualize how each
genome in each generation do, and how the other (lower candidates) does bad, and
what the survivors does good
- [ ] graphing capabilities, show the fitness function overtime while training

## To run the result
compile it directly with `cargo build --release`

Cross-compilation for windows is not yet tested or researched, please try with
caution
navigate to `/target/release`
run `./breakout-game` to play manually

run `./breakout-train` to train the AI
use `--num-gens <num>` to specify how many generations you want to make
use `--num-steps <num>` to specify how many steps each individual can make in
each evaluation

run `./breakout-ai` to visualize how the training results

## Postmortem
I initially intended to be written fully without using AI, but as time is
limited, there's still many parts that is written with AI. I only use Opencode with
Claude Sonnet 4.5 for this project.

The limitation that I decided for AI to work is to do the following:
- refactoring codes (game engine)
- gluing together all the different functions for training the population,
connecting it to the game engine
- debugging issues with the "rewritten" algorithms from Tech With Nikola's video
looking back, I think when rewriting the algorithm, I should also came up with
some ideas to test each of the functionality instead of just copy pasting a
bunch of them until eventually it is too much functions that I'm not sure they
can work enough, thus making it too hard to debug.

At the time when I started this project, I'm still working my way up in the Rust
Programming Book, in the Smart Pointers chapter, which feature I don't use in
this project.

Here's the stuff that I learned about Rust while making this project:
- function chains really is useful and convenient
- Rust release build is very powerful (my training process is faster by 100x!
Insane)
- I'm still getting the hang of visibility and encapsulation usage in this
language, unlike C++ private fields, in here, pub is used, and the many other
things like modules that I'm still a bit confused what for. So the practices to
structuring my code is still a bit sloppy.

My limitations in this project:
- I was too ambitious in writing an AI from scratch after seeing many videos of
  people doing it making it looks easy, as it turns out, I still need to learn a
  lot more.
  Particularly, I can't even write a simple AI for a single neuron, but what
about making a network that can evolve with cross-over and mutation? It was kind
of lucky that this project was on time and the model turns out quite good.
My next project should be a smaller or simpler version of the AI that I can 100%
write it myself without external assistance

- Not enough experience in game programming
eventhough breakout looks simple from the outside, I spent 1 afternoon only
implementing the ability of the platform to aim where the ball bounces, based on
the position of the collision of the ball 
I also found several bugs in the game, like the collision not being handled
properly, and then the weird bug where if the velocity of the ball is fast
enough it can teleport through the block, etc.
I may be able to solve these types of bugs, but a reasonable time is required.
But nonetheless though, it was fun. I wish the game was written by me, but it is
an example from macroquad library.

All in all, this project was really fun, and I really learned a lot. Before this
AI project, I've already did so many other "AI" projects that I honestly didn't
really care about, since at that time I don't have enough time to actually do
the project, nor do the project sounds interesting. Most projects written with
python hides too much complexity that I can't even grasp what am I dealing with.
Doing the basic things with system level language is the way to go, for me. Hope
this project helps with others that also wants to learn!
