"""The publication builder must be per-run and removed on build failure."""
import unittest
from unittest.mock import patch
import ghcr_image as ci


class BuilderIsolation(unittest.TestCase):
    @patch.object(ci, 'docker')
    def test_builder_is_explicit_and_removed(self, docker):
        env = {'DOCKER_CONFIG': '/scratch/publish-unique'}
        with ci.isolated_builder(env) as name:
            self.assertEqual(name, 'ci-publish-unique')
        self.assertEqual(docker.call_args_list[0].args[0], ['buildx', 'create', '--name', name, '--driver', 'docker-container'])
        self.assertEqual(docker.call_args_list[-1].args[0], ['buildx', 'rm', name])
        self.assertNotIn('--use', docker.call_args_list[0].args[0])

    @patch.object(ci, 'docker')
    def test_builder_cleanup_on_build_error(self, docker):
        with self.assertRaisesRegex(RuntimeError, 'build failed'):
            with ci.isolated_builder({'DOCKER_CONFIG': '/scratch/publish-failed'}):
                raise RuntimeError('build failed')
        self.assertEqual(docker.call_args.args[0], ['buildx', 'rm', 'ci-publish-failed'])
