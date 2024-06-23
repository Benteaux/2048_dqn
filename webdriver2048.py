from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import random
import time


actions = ["up", "down", "left", "right"]

driver = None



# Function to send a move command to the game
def send_move(move):
    body = driver.find_element(By.TAG_NAME, "body")
    if move == 'up':
        body.send_keys(Keys.ARROW_UP)
    elif move == 'down':
        body.send_keys(Keys.ARROW_DOWN)
    elif move == 'left':
        body.send_keys(Keys.ARROW_LEFT)
    elif move == 'right':
        body.send_keys(Keys.ARROW_RIGHT)

# Function to get the board state
def get_board():
    while True:
        try:
            board = [[0] * 4 for _ in range(4)]
            tiles = driver.find_elements(By.CLASS_NAME, "tile")
            for tile in tiles:
                # print("Tile get attribute class: ", tile.get_attribute("class"))
                tile_classes = tile.get_attribute("class").split()
                # print("Tile Classes: ", tile_classes)
                tile_value = int([cls.split("-")[1] for cls in tile_classes if cls.startswith("tile-") and not cls.startswith("tile-position-")][0])
                # print("Tile Value: ", tile_value)
                tile_position_class = [cls for cls in tile_classes if cls.startswith("tile-position-")][0]
                # print("Tile Position Class: ", tile_position_class)
                n, a, tile_x, tile_y = tile_position_class.split("-") # tile_position_class = "tile-position-x-y," and tile_position are unneeded
                tile_x, tile_y = int(tile_x), int(tile_y)
                board[tile_y - 1][tile_x - 1] = tile_value
            return board
        except Exception as e:
            print(f'Error occurred')
            time.sleep(0.02)

def show_board():
    board = get_board()
    for row in board:
        print(row)
    print("Sum: ", sum(sum(row) for row in board))

def get_score():
    score_element = driver.find_element(By.CLASS_NAME, "score-container")
    score = score_element.text.split("\n")[0]
    score = int(score)
    return score

def show_score():
    print("Score: ", get_score())


def show_state():
    show_board()
    show_score()
    print("\n")

def sample():
    i = random.randint(0, len(actions) - 1)
    return i

# send move is for non-ai purposes
def step(action_ix):
    send_move(actions[action_ix])
    time.sleep(0.02)
    return get_board(), get_score(), is_game_over()


def is_game_over():
    try:
        game_over_msg = driver.find_element(By.CLASS_NAME, "game-message.game-over")
        return True
    except:
        return False
    
def reset(isOver):
    new_game_button = driver.find_element(By.CLASS_NAME, "restart-button")
    new_game_button.click()
    if not isOver:
        time.sleep(0.03)
        alert = driver.switch_to.alert
        alert.accept()
    time.sleep(0.02)
    return get_board()


'''
# Get the current state of the board
show_state()

over = False
for _ in range(2000):
    send_move(sample())
    over = is_game_over()
    if over:
        print(_)
        reset(over)

# concern: if game over message takes a while to appear and process,
# ai will not know it has caused the game to end, and will likely
# wrongly attribute one of the next moves to causing the end of the game
reset(is_game_over())
show_state()



time.sleep(10) 
# Close the WebDriver
driver.quit()
'''