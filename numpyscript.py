# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Assignment 2019
# Programming for Data Analysis
# 
# %% [markdown]
# This contains all of the information for my 2019 submission for the Assignment for Programming for Data Analysis module.  Work on this began 13th October 2019 with a final submission date of November 11th.
# %% [markdown]
# NumPy, short for “Numerical Python” is a free to use, open source numerical library in python.  The library contains a large number of mathematical, algebriac and transformational functions.  Numpy contains a multi-dimensional array (more than one list, a list that can hold other lists, or lists within lists) and matrix data structures (numbers arranged in rows and columns).  Anaconda comes with NumPy preinstalled.  Otherwise it is necessary to install it via the pip install NumPy command line code.  Then import as np when using Python.  NumPy is utilised in data science and machine learning and to perform mathematical and statistical operations.  
# 
# NumPy. Random is a module used for random sampling in NumPy, providing a way of creating random samples with the NumPy.   It gives a variety of functions to generate random numbers.  
# 
# But the numbers may not be actually random, they just appear to be.  They may be pseudorandom or PRNG.  Pseudorandom number generators are algorithms that produce numbers that appear random, but are not really.  If the seed is reset each time (np.random.seed(0) ), the same set of numbers will always appear.  If random seed is not reset, different numbers will appear each time.   NumPy will generate a seed value from a part of your computer system such as the clock.
# 
# The random function can be used to build customized random number generators, it’s the key ingredient for many of the functions in the random module.  Random adds unpredictability to simulations such as the Monte Carlo simulation.  In a Monte Carlo simulation, a model is built showing the different outcomes that are probable.  The process can be complicated and not easy to predict due to random variables.  It is also known as multiple probability simulation.  Monte Carlo method was the first approach to use computer simulation for statistical problems. It was developed by John von Neumann, Stanisław Ulam, & Nicholas Metropolis, working on the Manhattan project during World War II.  
# 
# Random sampling can be broken down into 4 further sections, as per SciPy.org., which themselves then contain further subdivisions.  These are Simple random data, Permutations, Distributions and Random generator.
# 
# %% [markdown]
# The Simple Random Data generates random values, integers and floats which are dependant on the selected function.  Which function to use will be determined by the data used and what is the required output. 
# Rand will return values in a given shape from a predetermined parameter (uniform distribution). Here I have asked for random numbers in three columns and two rows. As a default, the numbers returned will be more than 0 but less than 1  But first NumPy has to be imported.
# 

# %%
import numpy as np


# %%
np.random.rand(3,2)

# %% [markdown]
# Here I have asked for 100 random numbers

# %%
x = np.random.rand (100)
x


# %%
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.hist(x)
plt.show()


# %%


# %% [markdown]
#  numpy.random.randn() return a sample (or samples) from a specific/standard normal distribution, the standard distribution.  It is  useful for adding random noise element into a dataset for initial testing of a machine learning model.  

# %%
x = np.random.randn (100)
x


# %%
plt.hist(x)
plt.show()

# %% [markdown]
# numpy.random.randint returns a random integer in the provided range, here I have given it 1 to 100 to select from.  If you were trying to simulate the throw of a dice, you would use (1, 6)

# %%
x = np.random.randint(1,100)
x


# %%
# A dice was rolled 10000 times and 6 was the largest number rolled

import scipy.stats
import sympy.stats
dice = scipy.stats.randint(1, 7)
dice.rvs(10000).max()

# %% [markdown]
# numpy.random.random_integers is similar to random_integers

# %%
np.random.random_integers(2)

# %% [markdown]
# Here I have asked for 3 columns and 2 rows of random numbers with 15 being the highest random number to choose from and starting from 1. 

# %%
np.random.random_integers(15, size=(3,2))

# %% [markdown]
# numpy.random.random_sample, numpy.random.random, numpy.random.ranf and numpy.random.sample are just different names for the same thing.
# 
# random_sample([size]) Return random floats in the half-open interval [0.0, 1.0).
# random([size]) Return random floats in the half-open interval [0.0, 1.0).
# ranf([size]) Return random floats in the half-open interval [0.0, 1.0).
# sample([size]) Return random floats in the half-open interval [0.0, 1.0).

# %%
np.random.random_sample


# %%
np.random.random


# %%
np.random.ranf


# %%
np.random.sample


# %%
np.random.random_sample is np.random.random


# %%
np.random.random_sample is np.random.ranf


# %%
np.random.random_sample is np.random.sample


# %%



# %%
np.random.random_sample()

# %% [markdown]
# The below script asks to select 4 columns with 2 rows of random numbers.

# %%
np.random.random_sample((4,2))

# %% [markdown]
# numpy.random.choice
# %% [markdown]
# Here I ask to select 4 random numbers up to 7.  The numbers that could be selected begin at 0 and go up to and include 6.

# %%
np.random.choice(7, 4)

# %% [markdown]
# To see what else it is capable of, here I script code to resemble a roulette wheel and the different possible outcomes.  Also included are the weights of the three colours which are used.  The wheel is spun 12 times and the results are printed to the screen.

# %%
import random
colours = ["Red", "Black", "Green"]
results = random.choices(colours, weights=[18, 18, 2], k=12)
print (results)

# %% [markdown]
# numpy.random.bytes returns random bytes

# %%
np.random.bytes(5)

# %% [markdown]
# And to take it a step further with bytes, an alternative way to generate a secure random sequence. Using the standard library secrets module for use in  passwords, account authentication, security tokens, and related secrets.

# %%
import secrets
b"\x11" + secrets.token_bytes(2) + b"\x10"

# %% [markdown]
# Permutations

# %%
np.random.permutation([0, 1, 0, 2, 5, 16])


# %%
arr = np.arange(16).reshape((8, 2))
np.random.permutation(arr)

# %% [markdown]
# takes a list as an input and return an object list of tuples that contain all permutation in a list form, it arranges the elements of an ordered list into a one on one mapping with itself.  
# 
# np.random.permutation has two differences from np.random.shuffle:
# 
# if passed an array, it will return a shuffled copy of the array; np.random.shuffle shuffles the array inplace
# if passed an integer, it will return a shuffled range i.e. np.random.shuffle(np.arange(n))  
# 
# np.random.permutation is useful when you need to shuffle ordered pairs, eg. for classification:
# %% [markdown]
# Create a table and take a look at it.

# %%
import pandas as pd
raw_data = {
    "country": ["England", "Austraila", "New Zeland",  "Wales", "France", "Japan", "South Africa", "Ireland"],
    "place": ["8", "7", "5", "5", "4", "3", "2", "1"]
    }
df = pd.DataFrame (raw_data,
                  index = pd.Index(["A", "B", "C", "D", "E", "F", "G", "H"]))
df


# %%



# %%



# %%
nrows = df.shape[0]
df

# %% [markdown]
# Randomise the sequence of the rows 

# %%
b = np.random.permutation(nrows)


# %%
b

# %% [markdown]
# Print out the data frame based on the array created above with permutation. Take function takes the indeces and change the order of the rows according to the indeces that we pass it.

# %%
df.take(b)

# %% [markdown]
# Randomise the index and then print out the table based on that.

# %%
df.index


# %%
np.random.permutation(df.index)

# %% [markdown]
# reindex function changes the order of the index. Its just another way of doing it.

# %%
df.reindex(np.random.permutation(df.index))

# %% [markdown]
# Sample method of randomisation.  Give me a random sample using all the rows

# %%
df.sample(n=nrows)


# %%



# %%
def randString(length=7):
    #Select letters to be used in the string
    letters='rebecca'
    return ''.join((random.choice(letters) for i in range(length)))
print('Password is =',  randString() )
print ('Password is =', randString(12) )   


# %%



# %%
# An example of a way to create a password using the permutation function from predefined letters in a preselected lenght.

Letters = np.array(['R', 'E', 'B', 'E', 'C', 'C', 'A',])
Order = np.random.permutation(Letters)
print('Generated password =', Order[2:12])

# %% [markdown]
# A different tool is used here, an itertool.  Iterators are data types that can be used in a for loop. The most common iterator in Python is the list. This doesn't relate really to the assignment but I went down a bit of a rabbit hole.

# %%



# %%
import itertools
a = [["Ireland", "England", "New Zeland"], ["Win", "Loose", "Draw"], [1, 2, 3]]
list(itertools.product(*a))


# %%
import numpy as np
np.random.permutation ([("Ireland", "England", "New Zeland"), ("Win", "Loose", "Draw"), (1, 2, 3)])


# %%



# %%
import numpy as np
loc, scale = 10, 1
s = np.random.logistic(loc, scale, 10000)
count, bins, ignored = plt.hist(s, bins=50)


# %%


# %% [markdown]
# Numpy.random.normal
# Enables the creation of an array in which the data looks to be distributed normally aka the bell curve or Gaussian distribution.   
# A list of random samples are selected from a normal distribution.  Different results will be drawn each time the function is executed 
# (this is due to the random part) but will generally be the same shape.  The peak is always in the middle and the curve is always symmetrical.  
# A larger size will give a better, more accurate result.  Bell curve provide a quick way to visualize the mean, mode and median.  
# If the distribution is normal the mean, median and mode are the same.
# 
# There are 3 key arguments which influence it.  Loc, scale and size.  The syntax will look like:  np.random.normal (loc=, scale=, size=)
#  np.random.normal is the function name, this is how we call it.
# Loc=,  sets the mean of the data, which defaults to 0 if no other value is chosen.  This is the very top of the curve.  In a normal distribution, fifty percent of the distribution lies to the right of the mean and fifty percent to the left.
#  
# 
# 
# Scale=, standard deviation of normal distribution, set to 1 as default.  Shows how much variation from the average.  
# If the data points were very close to the mean it shows a low standard deviation and if the data points were spread out it has a high standard deviation.  
# 
# ![bell%20curve%201.png](attachment:bell%20curve%201.png)
#  
# 
# The size controls the size and shape of the array.  This array can be modelled on a 1 dimensional or multi dimensional depending on the integers provided.  
# 
# ![bell%20curve.png](attachment:bell%20curve.png)

# %%
#Generate a single number from normal distribution. 

np.random.normal(size = 1, loc = 0, scale = 1)


# %%
np.random.normal(size = 5, loc = 0, scale = 1)


# %%
# We ask for 1000 numbers with a mean of 50.  The bigger the size gives a more accurate result.
np.random.seed(42)
df=np.random.normal(size = 1000, loc = 50)
print (df)


# %%
# We can see the mean is 50
df.mean()


# %%
np.random.seed(42)
np.random.normal(size = 1000, scale = 100).std()


# %%
np.random.seed(42)
np.random.normal(size = 1000, loc = 50, scale = 100)


# %%
# Plot a histogram of 1000 numbers with a mean of 50 and a standard deviation of 10
import seaborn as sns

plt.title("Example of normal distribution")
plt.show (sns.distplot(np.random.normal(50, 10.0, 1000), bins=20, color='b'))


# %%


# %% [markdown]
# numpy.random.exponential
# %% [markdown]
# The exponential is a continuous distribution describing the probability of an event occuring during a timeframe by looking at the the time or distance between the events.  It is often used in conjunction with a Poisson distribution which creates a model of the the number of occurrences in a fixed period of time, it is discrete and defined in integers.  There is a strong relationship between the two distributions which describe the same thing but from different perspectives.  
# 
# Non-negative numbers within an exponential distribution are memoryless.  The past has no impact on any future behaviour.  If a certain period of time was to pass before an event occurs, with memoryless, this would have no effect on when the event was to happen. eg. if a plane crashed, some people may think that the probability of it happening again to the same company would be greatly reduced but in reality, the chances if it happening at any given time have not changed.
# 
# Its probability density function is:  f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}), 
# 
# for x > 0 and 0 elsewhere. \beta is the scale parameter, which is the inverse of the rate parameter \lambda = 1/\beta. The rate parameter is an alternative, widely used parameterization of the exponential distribution 
# 
# An example of exponential could look at the number of times a website is accessed in a 30 minute period. We know it gets 180 views each hour so that is 90 views per 30 minutes. 
# lambada = 3 (90 views per 1/2 hour / 30 minutes = 3 views per minute)
# 

# %%
np.random.exponential(2, 8)


# %%
# 500 events, going in increments of 5

plt.show(plt.hist(np.random.exponential(5, 500), bins=40, color='B', alpha=.3))
plt.title('Exponential Distribution')
plt.xlabel('x variables')
plt.ylabel('f(x)')


# %%
lam = 3
x = np.arange(0, 10, .1)
y = lam * np.exp(- lam* x) 
plt.plot(x,y)
plt.title('Exponential: $\lambda$ =%.2f' % lam)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.show()


# %%
# an example of a standard poisson distribution, the event can only be measured as occurring or not as occurring
from scipy.stats import poisson
import seaborn as sb
data_binom = poisson.rvs(mu=4, size=1000)
ax = sb.distplot(data_binom,
                  kde=True,
                  color='blue',
                  hist_kws={"linewidth": 5,'alpha':1})
ax.set(xlabel='Poisson', ylabel='Occurance Freq')

# %% [markdown]
# numpy.random.multinomial is used when there are several possible outcomes.  It is similar to but different from Binomial, which has only two outcomes (success and failure). Eg. a die is rolled 10 times.  There are 6 possibilities (1, 2, 3, 4, 5, 6). That would be classed as a multinimial experiment.  If the die was rolled 10 times to see how often 2 comes up, it would be classed as a binomial experiment (2 = success, 1, 3, 4, 5,6 = failure)
# 
# 

# %%
# A dice is thrown 10 times, it landed on 1 once, 2 once, 3 twice, 4 three times, etc

np.random.multinomial(10, [1/6.]*6, size=1)


# %%
# Dice is now thrown 10 times and then 10 again 
# The first round of 10 it landed on 1 twice, 2 three times, 4 once etc. and in the second round of 10 it landed on 1 once, 2 twice, 3 zero times.

x=np.random.multinomial(10, [1/6.]*6, size=2)
print (x)


# %%



# %%



# %%
plt.show(plt.hist(x))


# %%
# just as an aside, while researching multinominal I came across this bit of code which seems to indicate that
# np.random.multinomial is faster than np.random.choice

print(np.__version__)
probs = [1/100.] * 100
get_ipython().run_line_magic('timeit', '-n 1 -r 1 [np.random.choice(100, p=probs) for x in range(10000)]')
get_ipython().run_line_magic('timeit', '-n 1 -r 1 [np.random.multinomial(1, probs).argmax() for x in range(10000)]')

# https://github.com/numpy/numpy/issues/7543


# %%


# %% [markdown]
# numpy.random.uniform is a type of probability distribution which each variable has the identical probability that it will be the outcome, all of the possible outcomes are equally likely. eg. a coin has an equal probability of getting heads or tails so it can be classed as having a uniform distribution. The same can be said about a deck of cards.  There are four suits and the chances of choosing either a diamond, heart, spade or club are the same.
# 
# There are 2 types of uniform distributions: continuous and discrete. With continuous each variable has an equal chance of being chosen and there can be a  infinite number of permutations.   
# 
# 
# Discrete distribution has a fixed list of possible values, it can not be subdivided and there may be a repeat of a value. Eg. with the throw of a dice it will always be between 1 and 6.

# %%


# %% [markdown]
# The random.random() function takes no parameters while random.uniform() takes two parameters, i.e., start and stop.
# The random.random() function generates a random float number between 0.0 to 1.0, but never returns 1.0.  random.uniform(start, stop) generates a random float number between the start (a) and stop (b) number.  Rounding may end up giving you b.

# %%



# %%
# random float from a uniform distribution between 0 and 1


print("Random float number is ", np.random.uniform())


# %%
# 6 random floats from  -5 to  5

np.random.uniform(-5.0, 5, 6)


# %%
print("Random float number with two decimal places is ", round(random.random(), 2))
print("Random float number between 5 and 30.5 with three decimal places is ", round(random.uniform(5,30.5), 3))


# %%



# %%
import numpy as np
import time
rang = 10000
tic = time.time()
for i in range(rang):
    sampl = np.random.uniform(low=0, high=5, size=(10))
print("it took: ", time.time() - tic)

tic = time.time()
for i in range(rang):
    ran_floats = [np.random.uniform(0,3) for _ in range(20)]
print("it took: ", time.time() - tic)


# %%
plt.hist(np.random.uniform(low=0.0, high=1.0, size=1000), bins=35, color='Y', alpha=1)
plt.xlabel("Probability")
plt.title("Uniform Distribution")
plt.show()

