import pandas as pd
import numpy as np
import random

def generate_video_session():
    # Focused: Long duration, minimal interaction
    # Noise: Some videos are short, some have more scrolling (comments)
    duration = random.randint(60, 1200) 
    scroll = random.randint(0, 20)       # Increased scroll range (reading comments)
    keys = random.randint(0, 15)         
    switches = random.randint(0, 5)      
    return [duration, scroll, keys, switches, 1] # 1 = Focused

def generate_reading_session():
    # Focused: Medium duration, steady scroll, low keys
    # Noise: Quick skims, or heavy note taking (more keys)
    duration = random.randint(60, 900)
    scroll = random.randint(10, 80)      
    keys = random.randint(0, 50)         # Taking notes?
    switches = random.randint(0, 5)      # Checking definitions
    return [duration, scroll, keys, switches, 1] # 1 = Focused

def generate_social_session():
    # Distracted: Short duration (rapid switching), high scroll, high keys
    # Noise: Sometimes you get stuck on one post (longer duration)
    duration = random.randint(5, 300)   
    scroll = random.randint(20, 100)     
    keys = random.randint(5, 150)       
    switches = random.randint(2, 30)     
    return [duration, scroll, keys, switches, 0] # 0 = Distracted

def generate_doom_scrolling_session():
    # Distracted: Medium duration (getting sucked in), High Scroll, Low Keys
    # Duration: 1 to 10 mins (60-600s)
    # Scroll: High (70-100%)
    # Keys: Low (0-10) - just scrolling
    duration = random.randint(30, 600)
    scroll = random.randint(70, 100)
    keys = random.randint(0, 10)
    switches = random.randint(0, 5)
    return [duration, scroll, keys, switches, 0] # 0 = Distracted

data = []
# Generate 200 samples of each scenario to have enough data
for _ in range(200):
    data.append(generate_video_session())
    data.append(generate_reading_session())
    data.append(generate_social_session())
    data.append(generate_doom_scrolling_session()) # Doom-scrolling retained

# Add some random noise samples
for _ in range(50):
    data.append([random.randint(0,1000), random.randint(0,100), random.randint(0,100), random.randint(0,20), random.randint(0,1)])

df = pd.DataFrame(data, columns=['duration', 'scroll_depth', 'key_count', 'switch_count', 'label'])
df.to_csv('synthetic_data.csv', index=False)
print(f"Generated synthetic_data.csv with {len(df)} samples.")
