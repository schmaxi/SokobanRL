import gymnasium as gym
from .sokoban_env import SokobanEnv
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np


class SokobanEnvQLearningSameLevel(SokobanEnv):
    metadata = {
        'render.modes': ['human'],
        'render_modes': ['human']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=500,
                 num_boxes=3,
                 num_gen_steps=None,
                 reset=True,
                 render_mode="human"):

        self.render_mode = render_mode
        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -1
        self.penalty_box_off_target = 0
        self.reward_box_on_target = 0
        self.reward_finished = 10
        self.reward_last = 0
        self.reward_dead_end = -5000

        self.dead_end_reached = False

        self.info = {}

        self.new_box_position = None
        self.old_box_position = None

        # Custom Variables
        self.no_action = 0
        self.moved_up_cnt = 0
        self.moved_down_cnt = 0
        self.moved_left_cnt = 0
        self.moved_right_cnt = 0
        self.moved_cnt = 0
        self.pushedBox_cnt = 0
        self.done_cnt = 0
        self.current_reward = 0
        self.start_env_set = False
        self.reward_last_episode = 0
        self.to_reset = True
        self.dead_end_cnt = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        # screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16) # TODO: change back
        screen_height, screen_width = (dim_room[0], dim_room[1])
        maxValue = dim_room[0] * dim_room[1]
        numberStates = num_boxes+1
        self.observation_space = Box(low=np.array([0] * numberStates), high=np.array([maxValue] * numberStates), shape=(numberStates,), dtype=np.int32)
        
        # a matrix where every field has a unique value so we can reduce the state space to num_boxes + 1 (player)
        self.positionMatrix = self.createPositionMatrix()

        # flag to initialize the positions array
        self.initialize_positions = True

        if reset:
            # Initialize Room
            _ = self.reset()

    def reset(self, seed=None, options= {"second_player":False,
                              "render_mode":'human',
                              "calledByUser":False}):
        second_player=options["second_player"]
        render_mode = options["render_mode"]
        calledByUser = options["calledByUser"]
        if calledByUser:
            self.initialize_positions = True

        if self.start_env_set == False:
            self.room_fixed_start, self.room_state_start, self.box_mapping_start = self.load_room()
            self.start_env_set = True

        self.room_fixed = self.room_fixed_start.copy()
        self.room_state = self.room_state_start.copy()
        self.box_mapping = self.box_mapping_start.copy()

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last_episode = self.current_reward
        self.reward_last = 0
        self.boxes_on_target = 0

        self.dead_end_reached = False

        self.to_reset = True

        starting_observation = self.render(render_mode)
        return starting_observation, self._get_info()

    # def step(self, action): #, observation_mode='rgb_array'):
    #     observation_mode = self.render_mode
    #     assert action in ACTION_LOOKUP
    #     assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

    #     if self.to_reset: # somehow in gymnasium if i reset this in reset(), tensorboard will not publish these stats
    #         # Custom Variables
    #         self.moved_cnt = 0
    #         self.pushedBox_cnt = 0
    #         self.current_reward = 0
    #         self.no_action = 0
    #     self.to_reset = False

    #     self.num_env_steps += 1

    #     self.new_box_position = None
    #     self.old_box_position = None

    #     moved_box = False

    #     if action == 0:
    #         moved_player = False

    #     # All push actions are in the range of [0, 3]
    #     elif action < 5:
    #         moved_player, moved_box = self._push(action)

    #     else:
    #         moved_player = self._move(action)

    #     self._calc_reward()
        
    #     terminated, truncated = self._check_if_done()

    #     # Convert the observation to RGB frame
    #     observation = self.render()

    #     self.info = {
    #         "action.name": ACTION_LOOKUP[action],
    #         "action.moved_player": moved_player,
    #         "action.moved_box": moved_box
    #     }
    #     self.updateCustoms()
    #     info = self._get_info()

    #     return observation, self.reward_last, terminated, truncated , info

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES


        if 'human' in mode:
            if self.initialize_positions:
                self.initialize_state()
                self.initialize_positions = False
            else:
                self.update_state()
            return np.array(self.state)


        else:
            super(SokobanEnvQLearningSameLevel, self).render(mode=mode)  # just raise an exception
        
    def update_state(self):
        # check which box was moved and update its position
        for i, pos in enumerate(self.state):
            if self.new_box_position is not None:
                if np.array_equal(pos, self.get1dPosition(self.old_box_position[0], self.old_box_position[1])):
                    self.state[i] = self.get1dPosition(self.new_box_position[0], self.new_box_position[1])
                    break
        # update player position
        playerPos = np.argwhere(self.room_state == 5).flatten()
        self.state[0] = self.get1dPosition(playerPos[0], playerPos[1])
        return np.array(self.state)

    def initialize_state(self):
        state = []
        # player position
        playerPos = np.argwhere(self.room_state == 5).flatten()
        state.append(self.get1dPosition(playerPos[0], playerPos[1]))
        box_positions =  np.argwhere(self.room_state == 4)
        for bp in box_positions:
            pos = bp.flatten()
            state.append(self.get1dPosition(pos[0], pos[1]))
        self.state = np.array(state)

    def get1dPosition(self,x,y):
        return self.positionMatrix[x][y]

    def createPositionMatrix(self):
        rows, cols = self.dim_room
        positions = []
        count = 0
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(np.int32(count))
                count += 1
            positions.append(row)
        return positions

    def load_room(self):
        room = ['##########',
        '##########',
        '#######  #',
        '###  # # #',
        '#     .  #',
        '# #      #',
        '# #####  #',
        '##### $$.#',
        '### @    #',
        '##########' ]
        

        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in room:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping

    def step(self, action): #, observation_mode='rgb_array'):
        observation_mode = self.render_mode
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw', 'human']

        if self.to_reset: # somehow in gymnasium if i reset this in reset(), tensorboard will not publish these stats
            # Custom Variables
            self.moved_cnt = 0
            self.pushedBox_cnt = 0
            self.current_reward = 0
            self.no_action = 0
        self.to_reset = False

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        terminated, truncated = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render()

        self.info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box
        }
        self.updateCustoms()
        info = self._get_info()
        # if done:
        #     info["maxsteps_used"] = self._check_if_maxsteps()
        #     info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
        
        return observation, self.reward_last, terminated, truncated , info

    def updateCustoms(self):
        # Custom Variables
        if self.info["action.name"] == "push down":
            self.moved_down_cnt += 1
        if self.info["action.name"] == "push up":
            self.moved_up_cnt += 1
        if self.info["action.name"] == "push left":
            self.moved_left_cnt += 1
        if self.info["action.name"] == "push right":
            self.moved_right_cnt += 1
        if self.info["action.name"] == "no operation":
            self.no_action += 1
        if self.info["action.moved_player"]:
            self.moved_cnt += 1
        if self.info["action.moved_box"]:
            self.pushedBox_cnt += 1
        if self._check_if_all_boxes_on_target():
            self.done_cnt += 1
        if self.dead_end_reached == True:
            self.dead_end_cnt += 1
        self.current_reward += self.reward_last


    # def _calc_reward(self):
    #     """
    #     Calculate Reward Based on
    #     :return:
    #     """
    #     # Every step a small penalty is given, This ensures
    #     # that short solutions have a higher reward.
    #     self.reward_last = self.penalty_for_step

    #     # count boxes off or on the target
    #     empty_targets = self.room_state == 2
    #     player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
    #     total_targets = empty_targets | player_on_target

    #     current_boxes_on_target = self.num_boxes - \
    #                               np.where(total_targets)[0].shape[0]

    #     # Add the reward if a box is pushed on the target and give a
    #     # penalty if a box is pushed off the target.
    #     if current_boxes_on_target > self.boxes_on_target:
    #         self.reward_last += self.reward_box_on_target
    #     elif current_boxes_on_target < self.boxes_on_target:
    #         self.reward_last += self.penalty_box_off_target
        
    #     game_won = self._check_if_all_boxes_on_target()        
    #     if game_won:
    #         self.reward_last += self.reward_finished

    #     self.dead_end_reached = self._check_if_dead_end()
    #     if self.dead_end_reached:
    #         self.reward_last += self.reward_dead_end

        
    #     self.boxes_on_target = current_boxes_on_target

    # def _check_if_done(self):
    #     # Check if the game is over either through reaching the maximum number
    #     # of available steps, by pushing all boxes on the targets or by reaching a dead end.        
    #     return self._check_if_all_boxes_on_target(), self._check_if_truncated()

    # def _check_if_all_boxes_on_target(self):
    #     empty_targets = self.room_state == 2
    #     player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
    #     are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
    #     return are_all_boxes_on_targets

    # def _check_if_truncated(self):
    #     return self._check_if_maxsteps() or self.dead_end_reached
    
    # # If a box is in a corner (that is not a target), that box cannot be moved anymore, therefore we want to terminate the current run
    # # Same goes for when two boxes are aligned next to a wall
    # def _check_if_dead_end(self): 
    #     return self._check_if_box_in_corner() or self._check_if_two_boxes_aligned_next_to_wall()

    # def _check_if_two_boxes_aligned_next_to_wall(self):
    #     box_positions =  np.argwhere(self.room_state == 4)
    #     for bp in box_positions:
    #         if self.room_state[bp[0]+1, bp[1]] == 4:
    #             if self.room_state[bp[0]+1, bp[1]+1] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
    #                 return True
    #             elif self.room_state[bp[0]+1, bp[1]-1] == 0 and self.room_state[bp[0], bp[1]-1] == 0: 
    #                 return True
    #         elif self.room_state[bp[0], bp[1]+1] == 4:
    #             if self.room_state[bp[0]+1, bp[1]+1] == 0 and self.room_state[bp[0]+1, bp[1]] == 0:
    #                 return True
    #             elif self.room_state[bp[0]-1, bp[1]+1] == 0 and self.room_state[bp[0]-1, bp[1]] == 0: 
    #                 return True
    #     return False

    # def _check_if_box_in_corner(self):
    #     box_positions =  np.argwhere(self.room_state == 4)
    #     for bp in box_positions:
    #         if self.room_state[bp[0]+1, bp[1]] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
    #             return True
    #         elif self.room_state[bp[0]+1, bp[1]] == 0 and self.room_state[bp[0], bp[1]-1] == 0:
    #             return True
    #         elif self.room_state[bp[0]-1, bp[1]] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
    #             return True
    #         elif self.room_state[bp[0]-1, bp[1]] == 0 and self.room_state[bp[0], bp[1]-1] == 0:
    #             return True
    #     return False


    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP
    


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
}


# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['human']