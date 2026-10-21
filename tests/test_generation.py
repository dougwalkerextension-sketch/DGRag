import unittest
from unittest.mock import Mock, patch

import torch
import local_generation as generation


class LocalGenerationTests(unittest.TestCase):
    def tearDown(self):
        generation.load_generator.cache_clear()

    def test_loads_cpu_models_from_cache_without_remote_code(self):
        module = Mock()
        with patch.dict('sys.modules', {'transformers': module}):
            generation.load_generator('test/model', 'pinned-revision')
        module.AutoTokenizer.from_pretrained.assert_called_once_with(
            'test/model', revision='pinned-revision', local_files_only=True, trust_remote_code=False)
        module.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            'test/model', revision='pinned-revision', local_files_only=True,
            trust_remote_code=False, dtype=torch.float32)
        module.AutoModelForCausalLM.from_pretrained.return_value.to.assert_called_once_with('cpu')

    def test_returns_only_generated_tokens_with_local_inference(self):
        tokenizer, model = Mock(), Mock()
        model.config.max_position_embeddings = 4096
        tokenizer.apply_chat_template.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        tokenizer.eos_token_id = 0
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer.decode.return_value = 'A local reply [1]'
        with patch('local_generation.load_generator', return_value=(tokenizer, model)):
            answer = generation.generate_answer([{'role': 'user', 'content': 'A clock?'}])
        self.assertEqual(answer, 'A local reply [1]')
        self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [4, 5])
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])

    def test_discards_old_history_before_current_evidence(self):
        tokenizer, model = Mock(), Mock()
        model.config.max_position_embeddings = 400
        def chat(messages, **kwargs):
            return {'input_ids': torch.ones((1, sum(len(m['content']) for m in messages)), dtype=torch.long)}
        tokenizer.apply_chat_template.side_effect = chat
        tokenizer.eos_token_id = 0
        model.generate.side_effect = lambda **kwargs: torch.cat([kwargs['input_ids'], torch.tensor([[9]])], dim=1)
        tokenizer.decode.return_value = 'A reply'
        messages = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'x' * 150},
                    {'role': 'assistant', 'content': 'y' * 150}, {'role': 'user', 'content': 'Current evidence'}]
        with patch('local_generation.load_generator', return_value=(tokenizer, model)), patch.dict('os.environ', {'GENERATION_MAX_NEW_TOKENS': '256'}):
            generation.generate_answer(messages)
        retained = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual([m['content'] for m in retained], ['Instructions', 'Current evidence'])


if __name__ == '__main__':
    unittest.main()
