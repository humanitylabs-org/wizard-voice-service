"""Read-only Linux image publication preflight; never logs in or pushes."""
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

from ghcr_image import clean_env, docker, isolated_builder


def main(image):
    with tempfile.TemporaryDirectory(prefix='forum-build-', dir=os.environ['RUNNER_TEMP']) as temp:
        env = clean_env(temp)
        archive = Path(temp, 'image.oci.tar')
        with isolated_builder(env) as builder:
            common = ['buildx', 'build', '--builder', builder, '--provenance=false',
                      '--build-arg', 'VCS_REF=' + os.environ['GITHUB_SHA']]
            docker([*common, '--platform', 'linux/amd64,linux/arm64', '--output', f'type=oci,dest={archive}', '.'], env)
            with tarfile.open(archive) as tar:
                def load_json(name):
                    member = tar.extractfile(name)
                    assert member is not None, name
                    return json.load(member)
                def load_descriptor(desc):
                    algorithm, digest = desc['digest'].split(':')
                    assert algorithm == 'sha256'
                    return load_json('blobs/sha256/' + digest)
                def architectures(index):
                    result = set()
                    for desc in index['manifests']:
                        obj = load_descriptor(desc)
                        if 'manifests' in obj:
                            result |= architectures(obj)
                        else:
                            config = load_descriptor(obj['config'])
                            assert config['os'] == 'linux'
                            result.add(config['architecture'])
                    return result
                found = architectures(load_json('index.json'))
                assert found == {'amd64', 'arm64'}, found
                print('MULTI_PLATFORM_OCI_OK', sorted(found), flush=True)
            docker([*common, '--platform', 'linux/amd64', '--load', '--tag', image, '.'], env)


if __name__ == '__main__':
    main(sys.argv[1])
