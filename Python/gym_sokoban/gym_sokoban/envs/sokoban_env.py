import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Space
from gymnasium.spaces import Box
import sys
sys.path.append("C:\\Users\\schnablm17\\Documents\\gym_sokoban")
sys.path.append("C:\\Users\\schnablm17\\Documents\\gym_sokoban\\gym_sokoban")
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np

class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=2,
                 num_gen_steps=None,
                 reset=True,
                 render_mode="rgb_array"):


        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        self.info = {}

        self.render_mode = render_mode

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_dead_end = -40 # needs to be higher than reward accumulated when just running around so the agent doesn't use the dead end to maximize its reward
        self.reward_last = 0

        self.dead_end_reached = False # own varibale so we don't have to loop over all the boxes twice (once in calc_reward() and once in check_if_done())

        # Custom Variables
        self.no_action = 0
        self.moved_up_cnt = 0
        self.moved_down_cnt = 0
        self.moved_left_cnt = 0
        self.moved_right_cnt = 0
        self.moved_cnt = 0
        self.pushedBox_cnt = 0
        self.moved_cnt = 0
        self.pushedBox_cnt = 0
        self.done_cnt = 0
        self.current_reward = 0
        self.reward_last_episode = 0
        self.to_reset = True
        self.dead_end_cnt = 0
        

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        # self.action_space = Discrete(len(ACTION_LOOKUP))
        self.action_space = Discrete(len(ACTION_LOOKUP))
        # screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16) # TODO: change back
        screen_height, screen_width = (dim_room[0], dim_room[1])
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            # Initialize Room
            _ = self.reset()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action): #, observation_mode='rgb_array'):
        observation_mode = self.render_mode
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

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

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished

        self.dead_end_reached = self._check_if_dead_end()
        if self.dead_end_reached:
            self.reward_last += self.reward_dead_end

        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps, by pushing all boxes on the targets or by reaching a dead end.        
        return self._check_if_all_boxes_on_target(), self._check_if_truncated()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_truncated(self):
        return self._check_if_maxsteps() or self.dead_end_reached
    
    # If a box is in a corner (that is not a target), that box cannot be moved anymore, therefore we want to terminate the current run
    # Same goes for when two boxes are aligned next to a wall
    def _check_if_dead_end(self): 
        return self._check_if_box_in_corner() or self._check_if_two_boxes_aligned_next_to_wall()

    def _check_if_two_boxes_aligned_next_to_wall(self):
        box_positions =  np.argwhere(self.room_state == 4)
        for bp in box_positions:
            if self.room_state[bp[0]+1, bp[1]] == 4:
                if self.room_state[bp[0]+1, bp[1]+1] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
                    return True
                elif self.room_state[bp[0]+1, bp[1]-1] == 0 and self.room_state[bp[0], bp[1]-1] == 0: 
                    return True
            elif self.room_state[bp[0], bp[1]+1] == 4:
                if self.room_state[bp[0]+1, bp[1]+1] == 0 and self.room_state[bp[0]+1, bp[1]] == 0:
                    return True
                elif self.room_state[bp[0]-1, bp[1]+1] == 0 and self.room_state[bp[0]-1, bp[1]] == 0: 
                    return True
        return False

    def _check_if_box_in_corner(self):
        box_positions =  np.argwhere(self.room_state == 4)
        for bp in box_positions:
            if self.room_state[bp[0]+1, bp[1]] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
                return True
            elif self.room_state[bp[0]+1, bp[1]] == 0 and self.room_state[bp[0], bp[1]-1] == 0:
                return True
            elif self.room_state[bp[0]-1, bp[1]] == 0 and self.room_state[bp[0], bp[1]+1] == 0:
                return True
            elif self.room_state[bp[0]-1, bp[1]] == 0 and self.room_state[bp[0], bp[1]-1] == 0:
                return True
        return False


    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, seed=None, options= {"second_player":False,
                              "render_mode":'rgb_array'}):
        second_player=options["second_player"]
        render_mode = options["render_mode"]
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(seed=None, options={"second_player":second_player, "render_mode":render_mode})

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last_episode = self.reward_last
        self.reward_last = 0
        self.boxes_on_target = 0

        self.dead_end_reached = False

        # Custom Variables
        self.to_reset = True

        starting_observation = self.render()
        return starting_observation, self._get_info()

    def render(self, close=None, scale=1):
        mode = self.render_mode
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in self.render_mode:
            from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in self.render_mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render()  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            # img = room_to_rgb(self.room_state, self.room_fixed) // uncomment for normal size
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    def _get_info(self):
        self.info["no_op_cnt"] = self.no_action
        self.info["pushed_box_cnt"] = self.pushedBox_cnt
        self.info["done_cnt"] = self.done_cnt
        self.info["current reward"] = self.current_reward
        self.info["reward_last_episode"] = self.reward_last_episode
        self.info["dead_end_cnt"] = self.dead_end_cnt
        return self.info


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

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
