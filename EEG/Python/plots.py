
baseline = 0.775 
names = [ "Random Forest", "Linear SVC", "Perceptron"]
#names = ["RF Meta Only", "Random Forest", "Linear SVC", "Perceptron"]
values = [0.871, 0.827, 0.829]
#values = np.array([0.873, 0.871, 0.827, 0.829])


x = range(len(values))

# split it up
#above_threshold = np.maximum(values - threshold, 0)
#below_threshold = np.minimum(values, threshold)

# and plot it
fig, ax = plt.subplots()
ax.bar(x, values, color="g")
#ax.bar(x, above_threshold, 0.35, color="r",
        #bottom=below_threshold)

plt.xticks(x, names, rotation = 70)

# horizontal line indicating the threshold
x = range(-1, 5)
ax.plot(x, [baseline, baseline, baseline, baseline, baseline, baseline], "k--", label = "Majority Vote")
plt.title("Held-out F1 Score Across Different Models")
plt.ylabel("F1 Score")
plt.xlabel("Classifier")
plt.legend()
plt.ylim(0,1)
plt.show()
