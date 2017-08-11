Abstraction in nature and intelligence
================================================================================
Abstraction is a very interesting concept in nature ranging from the molecules, to single cellular life, to human intelligence. 
[Abstraction is all encompassing](#abstraction-is-all-encompassing)
[Abstraction as an efficent state search](#abstraction-as-an-efficent-state-search)


<br>Abstraction is all encompassing
================================================================================
This section is an introduction to a way of thinking about all structure as abstraction.

### Levels of abstraction in nature
- Abstraction in molecules can be thought of as looking at molecules as a collection atomic and subatomic particles, combining them in a way that they naturally stay together. While there is inherent randomness in each of the parts (wave function) they act in a way that is very predictable at a higher level.

- In cells we see more of the same. Individual atoms or molecules may not be in the exact places they should be, but the macroscopic (to the cell) behaviour is very consistent and predicable.

- Even evolution works at a higher level of abstraction. Micro-changes through mutations shift the properties of animals; positive traits survive, negative ones die. At this high level, variations in individual cells make almost no difference. *The entire process of evolution works at a very high level of abstraction itself.*

### Abstraction in human thought
- Human intelligence seems to work the same way. Most interestingly is what this leads to. Science, storytelling, math. All of these are abstracting away the world as a model, then looking at possible actions in that model.

  - Mathematics is the most obvious of these, we take something in the real world, look at it, and build some way of mathematically explaining it. Say you have 5 sheep, a wolf eats 3, now you have 2 sheep. At a certain point we begin to build models that are built on those original models (axioms) which potentially have no meaning in the real world at all. 
    - This is actually a pretty good argument for [mathematical fictionalism](https://en.wikipedia.org/wiki/Philosophy_of_mathematics#Fictionalism) - [see below](#all-models-are-fiction). 
    - Note: math is the cleanest abstraction of the three. Once you accept the axioms, every abstraction at a higher level is a perfect simulation of all lower levels.


  - A scientific hypothesis is just another type of abstracted model. In biology or chemistry this makes sense, it's just applied physics... [XKCD](https://www.xkcd.com/435/). But what any type of science entails is taking something we observe, creating a model of the way we _think_ it will act, and testing if we were right. If we're right the model is good, if not then we change the model. (Tinfoil hat: What if there is no physics, or ground truth like string theory. Everything in the universe just happens and all of science is just more and more accurate models approximating it?)

  - Storytelling. The most curious type of model. At first glance this seems pretty different from the last two, there aren't any predictions that can be made from a simple story. But the way the abstraction is working is still the same. Take the real world, build some abstract world of what it could be, there are allowable actions based on that model. <br><br>_Sherlock Holmes: We found the location of the criminal! I'm going to throw a plate at the ground and go take a bath._<br><br> Only things that fit that abstract model of the world makes sense in the story, whether it be realistic, sci-fi, or comedy. To make a good story it should be logical within the bounds of the model, and that model itself should be logical within the bounds of how _other humans_ see the world. The way to tell if the story is good is see if it makes sense for other people and their models of the world, or at least the way they _want_ to see the world.

### All models are fiction
- TODO: Why all models are fiction.


<br>Abstraction as an efficient state search
================================================================================
This section is about some ways to use this idea of abstraction in ML.

### Base state space
The _base_ state space can be thought of something analogous to the real world. The base level of physics, (put on another tinfoil hat) if there is any.

In an example learning problem like a game of Pong, this would be the pixels on the screen. A very bad solution to Pong would be to just do simple RL with pixels as the input to the agent. Luckily the current state of RL research is doing much better.

### RL state abstractions
The classic Deep RL agent uses a Neural Network to modify its search space to something more manageble and much more useful than simple pixels. The new search space is an abstraction of the _base_ space, it represents the pixels in terms of learned features from a pre-trained neural network. 

Fancier versions like [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (add long term memory) or [Imagination Augmented RL](https://arxiv.org/abs/1707.06203) (RL agent picks from a plans with a series of actions vs sigle time-step actions directly) are essentially the same as classic Deep RL with tricks like using prior knowledge or reducing the possible paths to search. The abstraction used here is incredibly valuable because it shrinks the search space from something virtually impossible to calculate into something managable.

### Human abstractions
Unlike these RL agents, humans can have many layers of abstractions. We are able to creatively come up with new (creative) solutions to problems at a much higher level than a simple RL agent. A skilled civil engineer can just look at a building and tell if it will stand or fall. No calculus needed. 

At each level of learning skills humans first learn how do to something at a certain level of abstraction, for civil engineers this is calculate forces on simple structures like trusses. If a person sees enough examples at a certain level of abstraction, they begin to build intuition for how it works at that level. They usually don't even need to calculate anything, just tell you (_an approximation_ of) the answer immediately. 

These abstractions build on top of one another, first children learn addition, they use that model to learn subtraction, multiplication, & division. They use that in turn to learn simple functions. Then complex functions like cosines. Calculus. Fourier transforms. At each of the level of abstraction, they learn an intuitive model of how each abstraction works and use the intuition from each previous level to build an intuitive model for the next. 

Aside: [Chunking (psychology)](#aside-1)

### Type of abstraction
Using this way of thinking there are two main types of abstractions. Feed forward calculations and simulations.

- _Feed forward_ abstractions take in some input space, in classical Deep RL these are pixels. In higher level simulations this may be other variables which define something about the simulation. And they output something valuable like the contents of the image.
- _Simulation_ abstractions take the content in one space and make it into a simpler problem which can be easily used with a feed forward net.

### Testing simulation abstractions
In order for them to be useful, simulations must be an accurate model of the lower level, and require less computation. Even if the simulation is accurate for only certain cases (like Kirchhoff's ideal circuit laws vs electrical fields) it is still useful as long as you know when you can use it.

- The human brain does this all the time, it makes approximate predictions all the time implicitly in everyday life (overgeneralizing), when our model fails too often we modify it. We use this explicitly too. Arguably the greatest human "invention" is science. This process takes a hypothesis, tries it out in a lower level of abstraction (usually math or real world experiments), if it fits, then we accept the hypothesis, otherwise we reject it.
- In Machine Learning we can do the same thing. At each level of abstraction we test the simulation using the lower level of abstraction.
    - Note: This may be a clean abstraction like mathematics or a dirty one like Kirchhoff's laws.

### Accelerate creative search
If you accept the axiom that creativity is just finding an undiscovered point in the base state space that is meaningful (whatever that means), then you can use this idea of multi-layered abstractions to find new _ideas_ much quicker than just searching the base state space.

The reason this happens is because you can consider each simulation abstraction as a projection of the previous space into a new (simpler) space. This means a single step in the new space is a large change in the previous space. 

- Let's say we have an grid of pixels, an abstraction would be gradients, lines, or curves. A single step in the base space would be to change a single pixel, while a step in the abstracted space would be to add a new line, or change a gradient.
- If you abstract even further, you may be able to create a novel scene of characters because you know what a person looks like, and what animals, buildings, and cars look like.

When people think of new ideas in high-level complicated topics, they use their intuition (at the highest-level of abstraction) to figure out how something should work. Then they use a lower level of abstraction to show that their intuition is correct.

This is most prominent in math research, often great mathematicians say they discovered some idea because it _felt right_. These dicoveries just would not be possible without that very high level intuitive understanding. The intuition gives you the idea, lower level math proves that idea.
  
- [An example](https://math.byu.edu/~lzhao/ProfMath/ResearchIntuition.pdf)
- TODO: find more examples


<br>Experiments
================================================================================
### First experiment
- Create game where you need to shoot a cannon at a target. You get target coordinates, and action space is angle and velocity. Abstractions should let you be able to learn a mathematical simulation of the world to shoot at the target.


<br>Questions to consider
================================================================================
##### Two types of abstractions
- Humans obviously have a neural network getting things like visual features of an image or getting classes. How do you use both the absract simulation module and feature modules together?
  - Eg one way to abstract is to create an simpler, approximate model of a lower level (from calculating circuit behaviour using electrical fields vs Kirchhoff's laws). The other is to use a feedforward network to calculate features or classes.
  - It seems that this would need to be two seperate structures to be useful? 
    1. You take the highest level of abstracted simulation you can 
    2. Put a feedfoward net on top of that (like classical Deep RL)
    3. Search around that high level abstract space
    4. Use searches in this high level space to test if the abstracted simulation (generative model) is a good one by using the results to prune the space of the lower level model
        - (Eg try a sample problem of a circuit by calculating everything using electric fields and compare that to the results you get using Kirchhoff's laws)
    5. Tweak the abstracted simulation to match the results of the lower level model more closely
    6. Do this at **every** level of abstraction


<br>References and Notes
================================================================================
##### Aside 1
An interesting concept in human psychology is [chunking](https://en.wikipedia.org/wiki/Chunking_(psychology)), which allows for 4 chunks<!-- TODO find the source for this--> in memory at once. These chunks are seem to be analogous to the highest level of (intuitive) abstraction for the current problem.