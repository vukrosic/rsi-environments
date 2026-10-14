"""Real framework episodes with a deterministic LOCAL scripted code provider.

No model policy is trained. Native scorers execute real workloads. The provider
copies supplied starter code (baseline), which is compatibility evidence only.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import web
import verifiers.v1 as vf


async def run(args):
    requests = []
    async def completion(request):
        body = await request.json()
        observation = json.loads(body['messages'][-1]['content'])
        kind = ('kernel' if 'remaining_steps' in observation else
                'training' if 'api_version' in observation else 'inference')
        previous = sum(m['role'] == 'assistant' for m in body['messages'])
        # Inference diagnostic scoring necessarily ends with blocked/null.
        action_type = 'submit' if previous or kind == 'inference' else 'test'
        code = observation.get('starter_code') or observation.get('code')
        # Training feedback omits source, so recover the initial observation.
        if code is None:
            for message in body['messages']:
                if message['role'] == 'user':
                    try: initial = json.loads(message['content'])
                    except (ValueError, TypeError): continue
                    code = initial.get('starter_code') or initial.get('code')
                    if code: break
        content = json.dumps({'type': action_type, 'code': code})
        requests.append({'kind':kind, 'action':action_type, 'message_count':len(body['messages'])})
        return web.json_response({'id':f'local-{len(requests)}','object':'chat.completion',
            'created':int(time.time()), 'model':body['model'], 'choices':[
            {'index':0,'message':{'role':'assistant','content':content}, 'finish_reason':'stop'}],
            'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
    app = web.Application(client_max_size=16*1024*1024)
    app.router.add_post('/v1/chat/completions', completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = vf.resolve_env_config({'taskset':{'id':'rsi-coding-suite',
        'environments':args.environments, 'split':args.split},
        'agent':{'runtime':{'type':'subprocess'},
                 'harness':{'id':'rsi-coding-suite','output_dir':str(output / 'episodes'),
                            'compatibility_only':True,'horizon':2},
                 'timeout':{'rollout':1600}, 'network_notice':False},
        'timeout':{'episode':1700}, 'interception':{'type':'server'}})
    env = vf.load_environment(config)
    (output / 'framework-config.json').write_text(config.model_dump_json(indent=2))
    ctx = vf.ModelContext('local-scripted-starter', vf.EvalClientConfig(
        base_url=f'http://127.0.0.1:{port}/v1', api_key_var='RSI_LOCAL_UNUSED'))
    results = []
    try:
        async with env.serving():
            for task in env.taskset:
                episode = await env.run_episode(task, ctx)
                (output / f'{task.data.id}.json').write_text(episode.model_dump_json(indent=2))
                summary = {'task':task.data.id,'framework_traces':len(episode.traces),
                    'framework_calls':sum(len(t.calls) for t in episode.traces),
                    'framework_ok':episode.ok,'framework_errors':[e.message for e in episode.errors],
                    'native':[t.info.get('rsi') for t in episode.traces],
                    'trainable':[t.agent.trainable for t in episode.traces],
                    'scoring_errors':[[e.message for e in t.errors] for t in episode.traces]}
                results.append(summary)
                print(json.dumps(summary), flush=True)
    finally:
        await runner.cleanup()
    (output / 'summary.json').write_text(json.dumps({'results':results,'local_provider_requests':requests,
        'outer_policy_weights_trained':False,'learned_improvement_verified':False}, indent=2))
    if not all(r['framework_calls'] and r['native'] and r['native'][0].get('terminated') or
               r['framework_calls'] and r['native'][0].get('truncated') for r in results):
        raise RuntimeError('missing complete native rollout')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='suite/integration/evidence/smoke-20261014')
    parser.add_argument('--environments',nargs='+',default=['inference','training','kernel'])
    parser.add_argument('--split',choices=['train','heldout'],default='train')
    asyncio.run(run(parser.parse_args()))
