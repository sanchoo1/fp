import matplotlib.pyplot as plt

# Data
modes = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5']
average_frame_rates = [161.56, 32.54, 21.79, 1.62, 2.34]
average_interval_times = [0.0184, 0.0321, 0.0523, 0.9742, 0.4295]

# Plot average frame rates
plt.figure(figsize=(10, 5))
plt.bar(modes, average_frame_rates, color='blue', alpha=0.6)
plt.xlabel('Mode')
plt.ylabel('Average Frame Rate (FPS)')
plt.title('Average Frame Rate for Different Modes')
plt.show()

# Plot average interval times
plt.figure(figsize=(10, 5))
plt.bar(modes, average_interval_times, color='green', alpha=0.6)
plt.xlabel('Mode')
plt.ylabel('Average Interval Time (s)')
plt.title('Average Interval Time for Different Modes')
plt.show()
