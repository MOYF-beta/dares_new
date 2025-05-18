from exps.trainerMH_abc import TrainerMH

class TrainerMH(TrainerMH):
    def load_model(self):
        # 直接使用父类TrainerMH的load_model（已实现DARES_MH单模型多任务结构）
        super().load_model()