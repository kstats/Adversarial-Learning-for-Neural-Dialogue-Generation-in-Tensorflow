import pickle
import matplotlib.pyplot as plt

glob_steps = pickle.load(open("steps.p","rb"))
rew = pickle.load(open("rewards.p","rb"))
prep = pickle.load(open("perplexity.p","rb"))
dl = pickle.load(open("disc_loss.p","rb"))
gen_steps = pickle.load(open("gen_steps.p","rb"))
disc_steps = pickle.load(open("gen_steps.p","rb"))

#print steps
#print rew
#print prep
#print dl

ts = gen_steps

plt.plot(ts, rew, 'ro', label = "Generator's reward")
plt.plot(ts, dl, 'bo', label = "Discriminator's cost")
#plt.plot(ts, y2, 'go', label = "Generator's perplexity over time")
plt.legend(loc=5, borderaxespad=0.)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,1.2))
plt.ylim([0, 1.1])

plt.xlabel("Time step")
#plt.ylabel("Reward")
plt.show()