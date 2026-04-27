#!/usr/bin/env python3
"""Quick API connectivity test — explicit no-proxy httpx client."""
import os, time, httpx

# Create httpx client with NO proxy
http_client = httpx.Client(
    proxy=None,
    verify=True,
    timeout=httpx.Timeout(120.0),
    transport=httpx.HTTPTransport(proxy=None),
)

from openai import OpenAI

client = OpenAI(
    api_key='sk-fqzbxxcldzmedxdyilolegnkvllgbmauozxdmqmslarrvjkd',
    base_url='https://api.siliconflow.cn/v1',
    http_client=http_client,
)

# Test 1: LLM
system = 'You are a helpful assistant with memory. Answer based on context: user graduated with Business Administration from Stanford.'
t0 = time.time()
resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-4B',
    messages=[{'role': 'system', 'content': system},
              {'role': 'user', 'content': 'What degree did I graduate with?'}],
    temperature=0.0, max_tokens=256,
    extra_body={'enable_thinking': False},
)
print(f'LLM: {time.time()-t0:.1f}s | {repr(resp.choices[0].message.content)} | tokens={resp.usage.total_tokens}')

# Test 2: Embedding
t0 = time.time()
resp2 = client.embeddings.create(model='BAAI/bge-m3', input=['test embedding query'])
print(f'Embed: {time.time()-t0:.1f}s | dim={len(resp2.data[0].embedding)}')

print('\nAPI is working!')
