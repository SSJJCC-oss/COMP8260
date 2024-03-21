import turtle
import time
import random
import json
import os

# Game setup
delay = 0.01
score = 0
high_score = 0
wd = 850  # width
ht = 620  # height
total_games = 0
highest_reward_score = 0
highest_food_score = 0
pen = turtle.Turtle()
segments = []
offsetX = 120 #offset to shift the game area to the left


# Screen setup
wn = turtle.Screen()
wn.title("Snake Game")
wn.bgcolor("white")
wn.setup(width=wd, height=ht)
wn.tracer(0)

# Border setup
border = turtle.Turtle()
border.penup()  
border.goto(-300 - offsetX, 300)  
border.pendown()  
border.color("black")  
for _ in range(4): #draw boarder
    border.forward(600)  # The length of each side
    border.right(90)  
border.hideturtle() 

# Snake head setup
head = turtle.Turtle()
head.shape("square")
head.color("black")
head.penup()
head.goto(0 - offsetX, 0)
head.direction = "Stop"

# Food setup
food = turtle.Turtle()
colors = random.choice(['red', 'green', 'black'])
shapes = random.choice(['square', 'triangle', 'circle'])
food.speed(0)
food.shape(shapes)
food.color(colors)
food.penup()
food.goto(0 - offsetX, 100)

# Snake movement functions
def go_up():
    if head.direction != "down":
        head.direction = "up"

def go_down():
    if head.direction != "up":
        head.direction = "down"

def go_left():
    if head.direction != "right":
        head.direction = "left"

def go_right():
    if head.direction != "left":
        head.direction = "right"

def move_segments(): #function to move tail
    # Move each segment to the position of the segment in front of it
    for i in range(len(segments) - 1, 0, -1):
        x = segments[i-1].xcor()
        y = segments[i-1].ycor()
        segments[i].goto(x, y)
    # Move the first segment to where the head was, if there are segments
    if segments:
        x = head.xcor()
        y = head.ycor()
        segments[0].goto(x, y)

def move(): #moves head & segments together
    move_segments()#if tail found moves it to follow the ehad
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)
    elif head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)
    elif head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)
    elif head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)


def add_segment(): #adds segment to snakes tail
    global score
 # Create a new segment and position it
    new_segment = turtle.Turtle()
    new_segment.speed(0)
    new_segment.shape("square")
    new_segment.color("orange")  # Tail color
    new_segment.penup()
    if segments:  # If there are already segments in the snake
        last_segment = segments[-1]
        new_segment.goto(last_segment.position())  # Position new segment at the last segment's position
    else:  # If this is the first segment
        new_segment.goto(head.position())  # Position it at the head's position
    
    segments.append(new_segment)
    score += 10  # Update the score
    update_score_display() 

# snake collision detection
def handle_collision():
    global score, high_score, total_games, highest_reward_score, highest_food_score, exploration_rate
    
    # Update total games played and highest scores
    total_games += 1
    highest_reward_score = max(highest_reward_score, score)
    highest_food_score = max(highest_food_score, len(segments) * 10)  # 10 points per food
    
    exploration_rate = 1.0  # Reset exploration rate or adjust accordingly
    
    # Game over
    time.sleep(1)
    head.goto(0 - offsetX, 0)
    head.direction = "Stop"
    for segment in segments:
        segment.goto(1000, 1000)  # Move off-screen
    segments.clear()
    
    score = 0
    update_score_display() 
    reset_game()
      
#checks for dangers in the area
def is_danger_ahead(direction, head, segments): 
    next_position = head.position()
    if direction == "up":
        next_position = (head.xcor(), head.ycor() + 20)
    elif direction == "down":
        next_position = (head.xcor(), head.ycor() - 20)
    elif direction == "left":
        next_position = (head.xcor() - 20, head.ycor())
    elif direction == "right":
        next_position = (head.xcor() + 20, head.ycor())

    # Check for wall collision
    if next_position[0] >= wd/2 or next_position[0] <= -wd/2 or next_position[1] >= ht/2 or next_position[1] <= -ht/2:
        return True

    # Check for self collision
    for segment in segments:
        if segment.distance(next_position) < 20:
            return True

    return False 

#resets the game and adds tally to scores
def reset_game():
    global score, segments, total_games, highest_reward_score, highest_food_score
    # Update total games and highest scores
    total_games += 1
    highest_reward_score = max(highest_reward_score, score)
    highest_food_score = max(highest_food_score, len(segments) * 10) # 10 points per food
    
    # Reset the snake position and direction
    head.goto(0, 0)
    head.direction = "Stop"
    
    # Clear existing segments
    for segment in segments:
        segment.goto(1000, 1000)  # Move off-screen
    segments.clear()
    
    # Reset scores
    score = 0
    update_score_display()

# updates scores with tally
def update_score_display():
    pen.clear()
    # Position the scoreboard to the right of the game area
    start_x = 320 - offsetX  # Position the scoreboard to the right, with offset
    start_y = 270
    line_height = 30
    
    # Display the score information
    texts = [
        f"Score: {score}",
        f"High Reward Score: {highest_reward_score}",
        f"High Food Score: {highest_food_score}",
        f"Total Games: {total_games}",
    ]
    for i, text in enumerate(texts):
        pen.goto(start_x, start_y - i * line_height)
        pen.write(text, align="left", font=("candara", 18, "bold"))

# Score display setup before game starts
pen.speed(0)
pen.shape("square")
pen.color("black")
pen.penup()
pen.hideturtle()
update_score_display()

#### save functions ####
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))
# construct the full path to the Q-table file
q_table_path = os.path.join(script_dir, "q_table.json")
 
def load_q_table(filename="q_table.json"):
    try:
        with open(filename, "r") as file:
            q_table_str_keys = json.load(file)
        # Convert string keys back to tuples
        q_table = {eval(key): value for key, value in q_table_str_keys.items()}
        return q_table
    except FileNotFoundError:
        print(f"File {filename} not found. Starting with a new Q-table.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}. File might be malformed.")
        return {}

def save_q_table(q_table, filename="q_table.json"):
    with open(filename, "w") as file:
        # Convert tuple keys to strings
        q_table_str_keys = {str(key): value for key, value in q_table.items()}
        json.dump(q_table_str_keys, file)
        
#save game functions
def save_on_demand():
    save_q_table(q_table, q_table_path)

def quit_game():
    save_q_table(q_table, q_table_path)
    print("Q-table saved. Exiting game...")
    wn.bye() 
    
wn.listen()  # Start listening to keyboard events
wn.onkeypress(quit_game, "q") # Press 'q' to save & quit
wn.onkeypress(save_on_demand, "s")  # Press 's' to save  

        
# Q-learning setup
ACTIONS = ['up', 'down', 'left', 'right']
q_table = {}  # Initialize Q-table 
learning_rate = 0.1 # 
discount_factor = 0.99 # 
exploration_rate = 1.0 #
exploration_decay = 0.995  # How fast we reduce exploration rate
min_exploration_rate = 0.01 # the minimum rate the ai will reach after long play time


#find the current state of food and its postion to the snake head
def get_state(head, food, segments): 
    food_position = food.position()
    head_position = head.position()

    food_left = food_position[0] < head_position[0]
    food_above = food_position[1] > head_position[1]

    danger_straight = is_danger_ahead(head.direction, head, segments)
    danger_right = is_danger_ahead("right" if head.direction == "up" else "down" if head.direction == "right" else "up" if head.direction == "left" else "left", head, segments)
    danger_left = is_danger_ahead("left" if head.direction == "up" else "up" if head.direction == "right" else "down" if head.direction == "left" else "right", head, segments)

    state = (
        food_left,
        food_above,
        danger_straight,
        danger_right,
        danger_left
    )
    return state 

#chooses the enxt action based on qlearning principles
def choose_action(state):
    global exploration_rate, q_table
    
    # Initialize Q values for a state to 0 if the state is not already in the Q-table
    if state not in q_table:
        q_table[state] = {action: 0 for action in ACTIONS}
    
    # Exploration vs Exploitation: Decide whether to choose a random action or the best known action
    if random.uniform(0, 1) < exploration_rate:
        # Exploration: Choose a random action
        action = random.choice(ACTIONS)
    else:
        # Exploitation: Choose the best action based on the current Q-values
        # If multiple actions have the same Q-value, randomly select among them
        max_q_value = max(q_table[state].values())  # Find the max Q-value among actions for this state
        best_actions = [action for action, value in q_table[state].items() if value == max_q_value]  # List of actions with max Q-value
        action = random.choice(best_actions)  # Randomly select if multiple best actions
    
    return action

#update the table based on the actions results
def update_q_table(state, action, reward, next_state):
    # Check if the next_state is not in the Q-table and initialize if necessary
    if next_state not in q_table:
        q_table[next_state] = {a: 0 for a in ACTIONS}
    
    # Calculate the maximum Q-value for the actions in the next state
    next_max = max(q_table[next_state].values())
    
    # Update the Q-value for the state-action pair using the Q-learning formula
    old_value = q_table[state][action]
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    q_table[state][action] = new_value
 
#reward functions 
def calculate_distance(pos1, pos2): # calculate any distance between two points
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

def update_reward_for_food_distance(current_state, new_state, reward): # reward based on how close to food the snake got
    food_pos = food.position()  
    
    head_pos_current = (current_state[0], current_state[1])  # Extract head position from current_state
    head_pos_new = (new_state[0], new_state[1])  # Extract head position from new_state
    
    # Calculate distance to food before and after the move
    dist_before = calculate_distance(head_pos_current, food_pos)
    dist_after = calculate_distance(head_pos_new, food_pos)
    
    # Reward getting closer, penalize getting farther
    if dist_after < dist_before:
        reward += 0.1  
    else:
        reward -= 0.1  
    
    return reward

#rerds for each action
def calculate_reward(current_state, new_state, action):
    # Initialize a small negative reward for each move to encourage finding food quickly
    reward = -0.1
    
    # Check if the snake ate food
    if new_state == "food_eaten":  
        reward = 10  # Large positive reward for eating food

    # Check for collision with walls or self
    elif new_state == "collision":
        reward = -10  # Large negative reward for collision
    
    #adds reward based on food distance    
    reward = update_reward_for_food_distance(current_state, new_state, reward)
    
    #adds reward based on danger distance
    if is_danger_ahead:
        reward -= 5  # Penalize for moving towards danger

    return reward

#loads q table at the start
q_table = load_q_table(q_table_path)
      
# Main game loop---------------------------------------------
while True:
    wn.update()
    
    # Food is eatten
    if head.distance(food) < 20:
        x = random.randint(-270, 270)
        y = random.randint(-270, 270)
        food.goto(x - offsetX, y)  # with offset
        add_segment()

    # Collision detection logic
    if head.xcor() > (280 - offsetX) or head.xcor() < (-280- offsetX): # with offset
        handle_collision()
    if head.ycor() > 280  or head.ycor() < -280:
        handle_collision()
    
    # Get the current state
    current_state = get_state(head, food, segments)
    
    # Choose an action
    action = choose_action(current_state)

    # Execute the chosen action and moves snake head
    if action == 'up':
        go_up()
    elif action == 'down':
        go_down()
    elif action == 'left':
        go_left()
    elif action == 'right':
        go_right()
    move()
    
    # Calculates the new state and its rewards, update q_table and next state to choose
    new_state = get_state(head, food, segments)
    reward = calculate_reward(current_state, new_state, action)
    update_q_table(current_state, action, reward, new_state)
    current_state = new_state  

    time.sleep(delay)
    
    
    
    
    
    