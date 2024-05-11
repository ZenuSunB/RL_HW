from gym.envs.registration import register

register(
    id = "FightingiceEnv-v0",
    entry_point = "myenv.fightingice_env:FightingiceEnv"
)