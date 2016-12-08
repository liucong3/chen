require 'Data'
require 'Model1'
require 'config'

local function load_model(folder)
   local config = torch.load(folder .. '/config.t7')
   local model = nn.Model(config)
   local params = model:getParameters()
   local tio = require 'TensorIO'
   tio.loadTensor(folder .. '/best-param.txt', params)
   return model
end

local function loadVocab(filename, vocabSize)
   local vocab = {}
   local file = io.open(filename, 'r')
   for i = 1, vocabSize do
      local line = file:read()
      vocab[#vocab + 1] = line:split('\t')[1]
   end
   file:close()
   return vocab
end

local function printSentence(code, vocab)
   local ended = false
   local sentence = ''
   for i = 1, code:size(1) do
      if code[i] == 1 then
         sentence = sentence .. ' <s>'
      elseif code[i] == 2 then
         sentence = sentence .. ' </s>'
         ended = true
         break
      elseif code[i] >= #vocab then
         sentence = sentence .. ' <unk>'
      else
         sentence = sentence .. ' ' .. vocab[code[i] - 2]
      end
   end
   if not ended then 
      sentence = sentence .. ' ...'
   end
   print(sentence)
end

torch.setdefaulttensortype('torch.FloatTensor')

print(os.date() .. ' - Loading model ...')
local model = load_model(config.main.save)
model.config.is_training = false

print(os.date() .. ' -- Loading data ...')
local data = io.Data(model.config)
collectgarbage()

print(os.date() .. ' - Loading vocab ...')
local vocabSrc = loadVocab(config.main.src_vocab_file, config.train_data.src_vocab)
local vocabDst = loadVocab(config.main.dst_vocab_file, config.train_data.dst_vocab)

collectgarbage()
print(os.date() .. ' -- Testing ...')

local batch, labels = data:getBatch()
batch = batch:t():contiguous()
labels = labels:t():contiguous()
local predicted = model:forward(batch):contiguous()
for i = 1, model.config.batch_size do
   local src = batch:select(2, i):contiguous():view(-1)
   local dst = labels:select(2, i):contiguous():view(-1)
   local pred = predicted:select(2, i):contiguous():view(-1)
   printSentence(src, vocabSrc)
   printSentence(dst, vocabDst)
   printSentence(pred, vocabDst)
   print('')
end
