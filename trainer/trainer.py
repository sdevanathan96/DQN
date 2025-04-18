import math

import numpy as np
from utils.config import Config
from core.util import get_output_folder

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)

    def train(self, pre_fr=0):
        print("Starting Training....")
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state, _ = self.env.reset()
        for fr in range(pre_fr + 1, self.config.frames + 1):
            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0
            if self.agent.buffer.size() > self.config.batch_size:
                loss = self.agent.learning(fr)
                losses.append(loss)

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, np.mean(all_rewards[-10:]), loss, ep_num))

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state, _ = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')

class PPOTrainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        
        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)


    def train(self, pre_fr=0):
        print("Starting PPO Training....")
        losses = []
        all_rewards = []
        episodes_since_update = 0
        total_frames = 0
        total_frames_since_chk = 0
        
        while total_frames < self.config.frames:
            state, _ = self.env.reset()
            episode_reward = 0
            episode_done = False
            
            while not episode_done:
                action, action_prob, value = self.agent.act(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_done = terminated or truncated
                
                self.agent.buffer.add(state, action, action_prob, value, reward, episode_done)
                
                state = next_state
                episode_reward += reward
                total_frames += 1
                total_frames_since_chk += 1

                if total_frames % self.config.print_interval == 0:
                    recent_rewards = all_rewards[-10:] if all_rewards else [0]
                    print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % 
                            (total_frames, np.mean(recent_rewards), losses[-1], len(all_rewards)))
            
            episodes_since_update += 1
            all_rewards.append(episode_reward)
            
            if episodes_since_update >= self.config.episodes_per_update:
                loss = self.agent.learning(total_frames)
                losses.append(loss)
                episodes_since_update = 0

                if self.config.checkpoint and total_frames_since_chk >= self.config.checkpoint_interval:
                    self.agent.save_checkpoint(total_frames, self.outputdir)
                    total_frames_since_chk = 0
            
            if len(all_rewards) >= 100:
                avg_reward = float(np.mean(all_rewards[-100:]))
                if avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    print(f"Environment solved with avg reward {avg_reward:.2f} after {len(all_rewards)} episodes")
                    self.agent.save_model(self.outputdir, 'best')
                    if self.config.win_break:
                        break
        
        print(f"Training completed: {total_frames} frames, {len(all_rewards)} episodes")
        self.agent.save_model(self.outputdir, 'last')
        
        return all_rewards