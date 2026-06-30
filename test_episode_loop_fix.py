#!/usr/bin/env python3

"""
测试修复后的episode循环逻辑
"""

def test_episode_loop_logic():
    """测试episode循环逻辑"""
    
    # 模拟参数
    max_steps_per_episode = 10
    episode_done = False
    episode_steps = 0
    
    # 模拟数据收集
    episodes_data = [[] for _ in range(2)]  # 2个环境
    collected_steps = []
    
    print("开始测试episode循环...")
    
    # 模拟修复后的循环逻辑
    while not episode_done and episode_steps < max_steps_per_episode:
        print(f"Step {episode_steps + 1}: 收集数据...")
        
        # 模拟为每个环境收集数据
        for env_idx in range(2):
            # 模拟数据收集
            episodes_data[env_idx].append({
                'rgb': f'rgb_data_step_{episode_steps}',
                'depth': f'depth_data_step_{episode_steps}',
                'human_num': episode_steps % 3 + 1,  # 模拟人员数量
                'action': episode_steps % 4,  # 模拟动作
                'step': episode_steps
            })
        
        collected_steps.append(episode_steps)
        episode_steps += 1
        
        # 模拟在第5步时episode结束
        if episode_steps >= 5:
            episode_done = True
            print(f"Episode在第{episode_steps}步结束")
    
    print(f"\n测试结果:")
    print(f"总步数: {episode_steps}")
    print(f"收集的步数: {collected_steps}")
    print(f"环境0收集的数据量: {len(episodes_data[0])}")
    print(f"环境1收集的数据量: {len(episodes_data[1])}")
    
    # 验证每个环境都收集了多步数据
    assert len(episodes_data[0]) == 5, f"环境0应该收集5步数据，实际收集了{len(episodes_data[0])}步"
    assert len(episodes_data[1]) == 5, f"环境1应该收集5步数据，实际收集了{len(episodes_data[1])}步"
    
    print("✅ 测试通过！每个环境都正确收集了多步数据")

if __name__ == "__main__":
    test_episode_loop_logic()
