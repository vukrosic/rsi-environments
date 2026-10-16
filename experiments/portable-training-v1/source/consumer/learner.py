"""Remote CUDA Smol-Instruct LoRA learner, prepared before native consume.

No native scorer/store method is called here. step() never acquires a lease or
loads a model: the outer admitted worker must do that before entering consume.
"""
import importlib
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
import time

from journal import atomic,digest,native_digest,read,require,sha


class CudaLearner:
    def __init__(self,config,checkpoint=None,samples_path=None):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.config=config;self.samples_path=Path(samples_path) if samples_path else None
        require(torch.cuda.is_available(),'admitted CUDA worker required')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(min(1.0,4_500_000_000/torch.cuda.get_device_properties(0).total_memory))
        recipe=Path(config['recipe_root']).resolve()
        require(sha(recipe/'SOURCE-MANIFEST.json')=='f5e5625dee8a0cb3f60eb4d3437b409665b8eb8edf91dc62fac3a62e599dbf54',
                'exact old recipe utility source required, not its reward protocol')
        source_manifest=read(recipe/'SOURCE-MANIFEST.json')
        require(sha(recipe/'common.py')==source_manifest['files']['common.py']['sha256'],'file verification utility changed')
        spec=importlib.util.spec_from_file_location('frozen_recipe_file_utilities',recipe/'common.py')
        utilities=importlib.util.module_from_spec(spec);spec.loader.exec_module(utilities)
        require(sha(recipe/'recipe.py')==source_manifest['files']['recipe.py']['sha256'],'frozen recipe compiler changed')
        compiler_spec=importlib.util.spec_from_file_location('group8_closed_recipe',recipe/'recipe.py')
        self.compiler=importlib.util.module_from_spec(compiler_spec);compiler_spec.loader.exec_module(self.compiler)
        self.starter=(recipe/'starter-fixture.txt').read_text()
        require(sha(recipe/'ASSET-MANIFEST.json')=='02b6e0c1547283a267dbe05d40dbbf88aeff84f0474e7b00c5dc1f3a6e8a3b40',
                'exact Instruct model manifest required')
        model_manifest=read(recipe/'ASSET-MANIFEST.json')
        utilities.verify_files(config['model'],model_manifest['files'])
        require(config['bindings']['model_sha256']==model_manifest['files']['model.safetensors']['sha256'],
                'outer model file identity differs')
        helper_root=Path(config['grpo_helpers']).resolve()
        require(sha(helper_root/'grpo_pilot.py')=='b930cc047f6430f90ba91f4481fca993fa73a034ec38d1eda9c8ca28ab65d7a8'
                and sha(helper_root/'sample.py')=='3e5aab91fb22f05668fabea1e7a351ea3315028d019305cbd928cbb9417dfa07',
                'selected imported LoRA/sampling helpers changed')
        sys.path.insert(0,str(helper_root))
        self.helpers=importlib.import_module('grpo_pilot')
        require(Path(self.helpers.__file__).resolve()==helper_root/'grpo_pilot.py'
                and Path(sys.modules['sample'].__file__).resolve()==helper_root/'sample.py','wrong helper module origins')
        torch.manual_seed(41015)
        self.model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
            trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda')
        self.tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
        require(self.model.config.model_type=='llama' and self.model.config.hidden_size==576 and
                self.model.config.num_hidden_layers==30 and self.tokenizer.eos_token_id==2,'Smol135 Instruct identity required')
        self.params=self.helpers.attach(self.model,16,32)
        require(len(self.helpers.lora_named(self.model))==120 and all(p.dtype==torch.float32 for p in self.params),
                'rank16 q/v float32 LoRA required')
        self.optimizer=torch.optim.Adam(self.params,lr=1e-4)
        self.model.config.use_cache=False;self.model.eval()
        self.base_hash=self.helpers.digest(self.helpers.base_named(self.model))
        self.prepared_policy_state_sha256=None
        if checkpoint:
            require(sha(checkpoint['policy_state_path'])==checkpoint['policy_state_sha256'],'published policy blob changed')
            value=torch.load(checkpoint['policy_state_path'],map_location='cpu',weights_only=True)
            require(value['bindings']==config['bindings'] and value['base_sha256']==self.base_hash,'policy bindings/base differ')
            named=dict(self.helpers.lora_named(self.model))
            require(set(value['adapter'])==set(named),'adapter keys differ')
            with torch.no_grad():
                for name,param in named.items():
                    require(value['adapter'][name].shape==param.shape and value['adapter'][name].dtype==torch.float32,
                            'adapter tensor shape/type differs')
                    param.copy_(value['adapter'][name].to('cuda'))
            self.optimizer.load_state_dict(value['optimizer'])
            torch.set_rng_state(value['rng_cpu']);torch.cuda.set_rng_state_all(value['rng_cuda'])
            require(self.helpers.digest(self.helpers.lora_named(self.model))==checkpoint['adapter_weights_sha256'],
                    'loaded adapter tensors differ')
            self.prepared_policy_state_sha256=checkpoint['policy_state_sha256']

    def save_state(self,path):
        torch=self.torch
        def cpu(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu()
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu(v) for v in value)
            return value
        value=dict(bindings=self.config['bindings'],base_sha256=self.base_hash,
            adapter=cpu(dict(self.helpers.lora_named(self.model))),optimizer=cpu(self.optimizer.state_dict()),
            rng_cpu=torch.get_rng_state(),rng_cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()])
        with Path(path).open('xb') as stream:
            torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        self.prepared_policy_state_sha256=sha(path)
        return self.prepared_policy_state_sha256

    def initial_state(self,path):
        require(self.prepared_policy_state_sha256 is None,'initialization cannot reset existing lineage')
        checksum=self.save_state(path)
        return dict(policy_state_path=str(Path(path).resolve()),policy_state_sha256=checksum,
            base_sha256=self.base_hash,adapter_sha256=self.helpers.digest(self.helpers.lora_named(self.model)),
            optimizer_state_sha256=self.component_hash(self.optimizer.state_dict()),
            rng_state_sha256=self.component_hash(dict(cpu=self.torch.get_rng_state(),cuda=self.torch.cuda.get_rng_state_all())))

    def component_hash(self,value):
        """Canonical tensor/primitive digest, independent of torch.save bytes."""
        import json
        result=hashlib.sha256();torch=self.torch
        def visit(item):
            if isinstance(item,torch.Tensor):
                result.update(str(item.dtype).encode());result.update(str(tuple(item.shape)).encode())
                result.update(item.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item,dict):
                for key in sorted(item,key=lambda x:(type(x).__name__,str(x))):
                    result.update(repr(key).encode());visit(item[key])
            elif isinstance(item,(list,tuple)):
                result.update(type(item).__name__.encode())
                for child in item:visit(child)
            else:result.update(json.dumps(item,sort_keys=True,allow_nan=False).encode())
        visit(value);return result.hexdigest()

    def compiled_source(self,text):
        try:return self.compiler.compile_recipe(self.compiler.parse(text),self.starter)
        except self.compiler.RecipeError:
            return "# Closed recipe parse failure; candidate fault.\nraise SyntaxError('invalid recipe JSON')\n"

    def sample(self,prompt,output,seed,*,group_id,reset_checkpoint_sha256,sampled_policy_sha256):
        """Called outside native consume; explicit own sampler, no HF warpers."""
        torch=self.torch;tok=self.tokenizer
        require(self.prepared_policy_state_sha256 is not None,'published initial/checkpoint state required')
        require(sampled_policy_sha256==native_digest(dict(base=self.base_hash,
            adapter=self.helpers.digest(self.helpers.lora_named(self.model)),model=self.config['bindings']['policy_model_sha256'])),
            'declared sampled policy differs from actual loaded model tensors')
        ids=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,
            return_tensors='pt',return_dict=True)['input_ids'].to('cuda')
        require(1<=ids.shape[1]<=768,'complete bounded prompt required')
        rows=[]
        for slot in range(8):
            torch.manual_seed(seed+slot)
            tokens,lps=self.helpers.sample_group(self.model,ids,1,0.8,256,tok.eos_token_id)
            seq=tokens[0];end=(seq==tok.eos_token_id).nonzero()
            length=int(end[0])+1 if len(end) else len(seq);seq=seq[:length]
            decoded=seq[:-1] if int(seq[-1])==tok.eos_token_id else seq
            text=tok.decode(decoded,skip_special_tokens=False)
            rows.append(dict(slot=slot,tokens=seq.cpu().tolist(),logprobs=lps[0,:length].cpu().tolist(),
                             text=text,text_sha256=digest(dict(text=text)),source_text=self.compiled_source(text)))
        atomic(output,dict(schema='training-group8-samples-v1',group_id=group_id,
            reset_checkpoint_sha256=reset_checkpoint_sha256,sampled_policy_sha256=sampled_policy_sha256,
            sampler_seed=seed,prompt_ids=ids[0].cpu().tolist(),policy_state_sha256=self.prepared_policy_state_sha256,temperature=0.8,samples=rows))

    def step(self,plan,checkpoint,output_dir):
        torch=self.torch;started=time.monotonic()
        require(self.samples_path and sha(self.samples_path)==plan['samples_sha256'],'sealed sampled-token file changed')
        samples=read(self.samples_path)
        require(samples['schema']=='training-group8-samples-v1' and samples['group_id']==plan['group_id'] and len(samples['samples'])==8
                and samples['temperature']==0.8 and samples['policy_state_sha256']==checkpoint['policy_state_sha256'],
                'sampler/policy lineage differs')
        prompt_ids=samples['prompt_ids'];require(1<=len(prompt_ids)<=768,'bounded exact prompt required')
        counted=[];gap=0.0
        for slot,row in enumerate(samples['samples']):
            require(row['slot']==slot and 1<=len(row['tokens'])<=256 and len(row['tokens'])==len(row['logprobs']),
                    'sample token/slot layout differs')
            require(all(type(token) is int and 0<=token<self.model.config.vocab_size for token in row['tokens']),
                    'invalid sampled token id')
            decoded=row['tokens'][:-1] if row['tokens'][-1]==self.tokenizer.eos_token_id else row['tokens']
            require(self.tokenizer.decode(decoded,skip_special_tokens=False)==row['text'] and
                    digest(dict(text=row['text']))==row['text_sha256'] and self.compiled_source(row['text'])==row['source_text'],
                    'sampled tokens/text/closed compiler join differs')
            if 'candidate_source_hashes' in plan:
                require(hashlib.sha256(row['source_text'].encode()).hexdigest()==plan['candidate_source_hashes'][slot],
                        'native candidate differs from sampled recipe source')
            if plan['advantages'][slot] is None:continue
            full=torch.tensor(prompt_ids+row['tokens'],device='cuda',dtype=torch.long)
            old=torch.tensor(row['logprobs'],device='cuda',dtype=torch.float32)
            require(bool(torch.isfinite(old).all()) and bool((old<=0).all()),'invalid own sampler likelihood')
            with torch.no_grad():
                replay=self.helpers.token_logprobs(self.model,full,len(prompt_ids),0.8)
                gap=max(gap,float((replay-old).abs().max()))
            counted.append((slot,full,old,plan['advantages'][slot]))
        require(len(counted)==plan['counted']>=2 and gap<=1e-4,'own sampler/replay gap exceeds1e-4 before optimizer')
        before=self.helpers.digest(self.helpers.lora_named(self.model));tokens=sum(len(old) for _,_,old,_ in counted)
        self.optimizer.zero_grad(set_to_none=True);loss_before=0.0
        for _,full,old,advantage in counted:
            lp=self.helpers.token_logprobs(self.model,full,len(prompt_ids),0.8)
            ratio=torch.exp(lp-old);objective=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
            loss=-objective.sum()/tokens;require(bool(torch.isfinite(loss)),'nonfinite GRPO loss')
            loss.backward();loss_before+=float(loss.detach())
            del lp,ratio,objective,loss
        grad_norm=float(torch.nn.utils.clip_grad_norm_(self.params,1.0,error_if_nonfinite=True))
        self.optimizer.step();torch.cuda.synchronize()
        loss_after=0.0
        with torch.no_grad():
            for _,full,old,advantage in counted:
                lp=self.helpers.token_logprobs(self.model,full,len(prompt_ids),0.8)
                ratio=torch.exp(lp-old)
                loss_after+=float(-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).sum()/tokens)
        require(self.helpers.digest(self.helpers.base_named(self.model))==self.base_hash,'frozen base changed')
        require(all(bool(torch.isfinite(p).all()) for p in self.params),'nonfinite adapter')
        policy=Path(output_dir)/'policy-adam-rng.pt';state_hash=self.save_state(policy)
        return dict(schema='training-researcher-group8-step-v1',runtime_kind='cuda_lora',group_id=plan['group_id'],
            plan_sha256=digest(plan),bindings=self.config['bindings'],optimizer_steps=1,
            previous_policy_state_sha256=checkpoint['policy_state_sha256'],policy_state_path=str(policy.resolve()),
            policy_state_sha256=state_hash,base_before_sha256=self.base_hash,base_after_sha256=self.base_hash,
            adapter_before_sha256=before,adapter_after_sha256=self.helpers.digest(self.helpers.lora_named(self.model)),
            loss=loss_before,loss_before=loss_before,loss_after=loss_after,loss_kind='grpo_ppo_surrogate',
            optimizer_state_sha256=self.component_hash(self.optimizer.state_dict()),
            rng_state_sha256=self.component_hash(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),
            max_logprob_gap=gap,grad_norm=grad_norm,wall_seconds=time.monotonic()-started,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
