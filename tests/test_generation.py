import unittest
from unittest.mock import Mock, patch

import local_generation as generation


class LocalGenerationTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{'role': 'system', 'content': 'Instructions'},
                         {'role': 'user', 'content': 'Earlier question ' * 1000},
                         {'role': 'user', 'content': 'Current question and evidence'}]

    def test_uses_local_deepseek_and_returns_only_final_answer(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'message': {'thinking': 'Private intermediate reasoning',
                                                 'content': 'A final answer [1]'}}
        with patch('local_generation.requests.post', return_value=response) as post:
            answer = generation.generate_answer(self.messages)
        self.assertEqual(answer, 'A final answer [1]')
        self.assertEqual(post.call_args.args[0], 'http://127.0.0.1:11434/api/chat')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], generation.generation_settings()['model'])
        self.assertTrue(payload['think'])
        self.assertEqual(payload['messages'][0], self.messages[0])
        self.assertEqual(payload['messages'][-1], self.messages[-1])
        self.assertEqual(len(payload['messages'][1]['content']), 1000)

    def test_strips_legacy_thinking_tags(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'message': {'content': '<think>Intermediate reasoning</think>Final answer.'}}
        with patch('local_generation.requests.post', return_value=response):
            self.assertEqual(generation.generate_answer(self.messages), 'Final answer.')

    def test_reasoning_without_final_answer_reports_token_budget(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'message': {'thinking': 'Intermediate reasoning', 'content': ''}}
        with patch('local_generation.requests.post', return_value=response):
            with self.assertRaisesRegex(ValueError, 'DEEPSEEK_MAX_NEW_TOKENS'):
                generation.generate_answer(self.messages)

    def test_missing_model_and_connection_errors_have_recovery_instructions(self):
        with patch('local_generation.requests.post', return_value=Mock(status_code=404)):
            with self.assertRaisesRegex(ValueError, 'ollama pull'):
                generation.generate_answer(self.messages)
        with patch('local_generation.requests.post', side_effect=generation.requests.ConnectionError('private')):
            with self.assertRaisesRegex(ValueError, 'Start Ollama') as error:
                generation.generate_answer(self.messages)
        self.assertNotIn('private', str(error.exception))

    def test_ready_checks_installed_model_without_hosted_requests(self):
        response = Mock()
        response.json.return_value = {'models': [{'name': generation.generation_settings()['model']}]}
        with patch('local_generation.requests.get', return_value=response) as get:
            self.assertTrue(generation.generation_ready())
        self.assertEqual(get.call_args.args[0], 'http://127.0.0.1:11434/api/tags')
        with patch('local_generation.requests.get', side_effect=generation.requests.ConnectionError):
            self.assertFalse(generation.generation_ready())


if __name__ == '__main__':
    unittest.main()
