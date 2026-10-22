import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import model_comparison as comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.matches = [{'metadata': {'name': 'THE BLACK CAT - Volume 2', 'text': 'A quoted passage.'}}]
        self.messages = [{'role': 'system', 'content': 'Use evidence.'},
                         {'role': 'user', 'content': 'Question and evidence.'}]

    def test_missing_env_key_is_not_connected_and_never_calls_openai(self):
        with TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENAI_API_KEY': 'inherited-key'}):
            settings = comparison.openai_config(Path(directory))
            with patch('model_comparison.requests.post') as post:
                outputs = comparison.compare_models(self.messages, self.matches, lambda _: 'Local reply',
                                                    'Qwen', 'MiniLM', settings)
            post.assert_not_called()
        self.assertEqual([o['id'] for o in outputs], ['local', 'minilm', 'openai'])
        self.assertEqual(outputs[-1]['text'], 'not connected')
        self.assertEqual(outputs[-1]['status'], 'not_connected')
        self.assertIn('A quoted passage.', outputs[1]['text'])

    def test_local_and_openai_execute_concurrently_and_emit_independently(self):
        rendezvous = threading.Barrier(2)
        release = threading.Event()
        streamed = []
        def local(messages):
            rendezvous.wait(timeout=5)
            self.assertTrue(release.wait(timeout=5))
            return 'Local reply'
        def remote(messages, settings):
            rendezvous.wait(timeout=5)
            return 'OpenAI reply'
        def emit(item):
            streamed.append(item['output']['id'])
            if item['output']['id'] == 'openai':
                release.set()
        with patch('model_comparison.openai_answer', side_effect=remote):
            outputs = comparison.compare_models(self.messages, self.matches, local, 'Qwen', 'MiniLM',
                                                {'key': 'test-key', 'model': 'test-model'}, emit=emit)
        self.assertTrue(all(o['status'] == 'complete' for o in outputs))
        self.assertLess(streamed.index('openai'), streamed.index('local'))

    def test_local_failure_does_not_hide_other_outputs_or_leak_exception(self):
        with patch('model_comparison.openai_answer', return_value='Remote reply'):
            outputs = comparison.compare_models(self.messages, self.matches,
                Mock(side_effect=RuntimeError('secret-key')), 'Qwen', 'MiniLM',
                {'key': 'test-key', 'model': 'test-model'})
        self.assertEqual(outputs[0]['status'], 'error')
        self.assertNotIn('secret-key', outputs[0]['text'])
        self.assertEqual(outputs[1]['status'], 'complete')
        self.assertEqual(outputs[2]['text'], 'Remote reply')

    def test_openai_uses_responses_api_and_separate_history(self):
        response = Mock(ok=True)
        response.json.return_value = {'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': 'Remote answer [1]'}]}]}
        with patch('model_comparison.requests.post', return_value=response) as post:
            text = comparison.openai_answer(self.messages, {'key': 'test-key', 'model': 'configured-model'})
        self.assertEqual(text, 'Remote answer [1]')
        self.assertEqual(post.call_args.args[0], 'https://api.openai.com/v1/responses')
        payload = post.call_args.kwargs['json']
        self.assertFalse(payload['store'])
        self.assertEqual(payload['model'], 'configured-model')
        self.assertEqual(payload['input'], self.messages[1:])

    def test_openai_errors_do_not_disclose_provider_body(self):
        response = Mock(ok=False, status_code=401, text='private secret diagnostic')
        with patch('model_comparison.requests.post', return_value=response):
            with self.assertRaisesRegex(ValueError, 'HTTP 401') as error:
                comparison.openai_answer(self.messages, {'key': 'test-key', 'model': 'test-model'})
        self.assertNotIn('private', str(error.exception))

    def test_openai_failure_does_not_hide_local_and_minilm(self):
        with patch('model_comparison.openai_answer', side_effect=ValueError('OpenAI timed out. Try again.')):
            outputs = comparison.compare_models(self.messages, self.matches, lambda _: 'Local reply',
                'Qwen', 'MiniLM', {'key': 'test-key', 'model': 'test-model'})
        self.assertEqual([o['status'] for o in outputs], ['complete', 'complete', 'error'])
        self.assertEqual(outputs[0]['text'], 'Local reply')
        self.assertIn('A quoted passage.', outputs[1]['text'])


if __name__ == '__main__':
    unittest.main()
