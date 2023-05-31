

@register_llm_trainer("reward")
class RewardTrainer(Trainer):
    def __init__(...)
        super().__init__(...)
        self.chosen_key = config['reward']['chosen_key']
        self.rejected_key = config['reward']['rejected_key']
        self.reward_key = config['reward']['reward_key']

    def train_step(
        self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], should_step: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.chosen_key not in inputs and self.rejected_key not in inputs:
            raise ValueError('RLHFTrainer requires chosen and rejected inputs')
        
        outputs_j = self.model({self.chosen_key: inputs[self.chosen_key]})
        outputs_k = self.model({self.rejected_key: inputs[self.rejected_key]})
        
        loss = self.rlhf_loss(outputs_j[self.reward_key][LOGITS], outputs_k[self.reward_key][LOGITS])
        all_losses = loss
        
        return loss, all_losses