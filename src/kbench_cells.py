# Cells typed verbatim into the Kaggle Benchmarks task notebook (new-benchmark-task-33547).
# Written as one-liners on purpose: the editor auto-indents, and exec()-ing code there kills the session.
# Proxy calls must run in a cell's own code; outputs land in /kaggle/working.

# --- CELL A: generate host-style prompts (domain mix follows the reconstructed train prompts) ---
CELL_A = r'''import kaggle_benchmarks as kbench, pandas as pd, re
G=next(k for k in kbench.llms if 'claude-sonnet-5@' in k); D={'physics':32,'biology':28,'philosophy':21,'creative prose (short stories, vignettes, imaginative descriptions)':16,'mathematics':15,'history':12,'computer science':10,'psychology':6,'economics':5,'linguistics, art or geography':5}
Q="Write {n} distinct prompts in the domain of {d} that someone might give to an AI assistant. Each asks for a short vivid piece of writing about one specific named thing (a named effect, paradox, experiment, organism, event, theorem, thought experiment, place or person), e.g. 'Explain the Jevons paradox with an everyday analogy.' or 'Describe standing on the surface of a neutron star.' Mix types: mostly explain-a-concept, many describe or narrate, some argue a position, a few short stories. Vary style modifiers (vivid, poetic, haunting, whimsical, for a general audience, use an everyday analogy) and leave some plain. One sentence each. Output only the prompts, one per line, no numbering."; P=[]
for d,w in D.items(): t=str(kbench.llms[G].prompt(Q.format(n=round(w*2.2),d=d))); L=[re.sub(r'^[\s\d.)*-]+','',x).strip() for x in t.split('\n')]; L=[x for x in L if len(x)>20]; P+=[(d,x) for x in L]; print(d,len(L),flush=True)
df=pd.DataFrame(P,columns=['domain','prompt']).drop_duplicates('prompt'); df.to_csv('prompts.csv',index=False); print('TOTAL',len(df))'''

# --- CELL B: generate passages, 4 workers, 150s timeout per call, resumable, saves every row ---
# Cost observed in the pilot: ~$0.015/passage; daily quota $10 -> keep LIMIT around 600 per day.
CELL_B = r'''import kaggle_benchmarks as kbench, pandas as pd, os, random, threading, glob; from concurrent.futures import ThreadPoolExecutor, as_completed
M={'gpt-5.5':'openai/gpt-5.5-','gpt-5.6-luna':'openai/gpt-5.6-luna','gpt-5.6-sol':'openai/gpt-5.6-sol','gpt-5.6-terra':'openai/gpt-5.6-terra','gpt-6-astra':'openai/gpt-6-astra','gemini-3.1-pro-preview':'google/gemini-3.1-pro-preview','gemini-3.6-flash':'google/gemini-3.6-flash','gemini-3.8-flash':'google/gemini-3.8-flash','claude-opus-4-6':'anthropic/claude-opus-4-6','claude-opus-4-8':'anthropic/claude-opus-4-8','claude-opus-5':'anthropic/claude-opus-5@','claude-sonnet-5':'anthropic/claude-sonnet-5@'}; K={l:next(k for k in kbench.llms if s in k) for l,s in M.items()}
WL=pd.read_csv(glob.glob('/kaggle/input/**/train.csv',recursive=True)[0]).text.str.split().str.len().tolist(); T="{p}\n\nWrite your answer as a single self-contained paragraph of about {n} words, with no headings, lists or markdown."; PR=pd.read_csv('prompts.csv').prompt.tolist(); OUT='gen_data.csv'; random.seed(1); J=[(f'gen_{i}',p,l,random.choice(WL)) for i,p in enumerate(PR) for l in M]; done=set()
if os.path.exists(OUT): x=pd.read_csv(OUT); x=x[~x.text.astype(str).str.startswith('ERROR')]; done=set(zip(x.prompt_id,x.model))
LIMIT=480; J=[j for j in J if (j[0],j[2]) not in done][:LIMIT]; print(len(J),'jobs',flush=True)
def call(j): b={}; th=threading.Thread(target=lambda: b.update(t=str(kbench.llms[K[j[2]]].prompt(T.format(p=j[1],n=j[3])))), daemon=True); th.start(); th.join(150); return dict(prompt_id=j[0],model=j[2],target_words=j[3],prompt=j[1],text=b.get('t','ERROR: timeout or failure'))
ex=ThreadPoolExecutor(4); fs=[ex.submit(call,j) for j in J]
for c,f in enumerate(as_completed(fs)): r=f.result(); pd.DataFrame([r]).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False); print(c,r['model'],r['target_words'],len(r['text'].split()),flush=True)'''
