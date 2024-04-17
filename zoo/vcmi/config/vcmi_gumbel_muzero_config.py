from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 30
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

vcmi_gumbel_muzero_config = dict(
    exp_name=f'data_mz_ctree/vcmi_gumbel_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            # observation_shape=(11, 15, 86),
            observation_shape=(86, 11, 15),
            action_space_size=2311,
            image_channel=86,  # does not seem to work
            num_res_blocks=3,
            num_channels=32,
            fc_reward_layers=[256],
            fc_value_layers=[256],
            fc_policy_layers=[256],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
            norm_type='BN',
            downsample=False,
            discrete_action_encoding_type='not_one_hot',
        ),
        cuda=False,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=5,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        # grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        max_num_considered_actions=10,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=30,
        num_unroll_steps=3,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(500),
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
vcmi_gumbel_muzero_config = EasyDict(vcmi_gumbel_muzero_config)
main_config = vcmi_gumbel_muzero_config

vcmi_gumbel_muzero_create_config = dict(
    env=dict(
        type='vcmi_lightzero',
        import_names=['zoo.vcmi.envs.vcmi_lightzero_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='gumbel_muzero',
        import_names=['lzero.policy.gumbel_muzero'],
    ),
)
vcmi_gumbel_muzero_create_config = EasyDict(vcmi_gumbel_muzero_create_config)
create_config = vcmi_gumbel_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
