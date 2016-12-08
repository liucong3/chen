
-- The namespace
config = {}

-- Training data
config.train_data = {}
config.train_data.file =  paths.concat(paths.cwd(), "../chen.code.txt")
config.train_data.batch_size = 16 --128

config.train_data.input_size = 25
config.train_data.output_size = 50

config.train_data.src_vocab = 50000 --50000
config.train_data.dst_vocab = 5000 --5000
config.train_data.src_emb = 300 --1000
config.train_data.dst_emb = 300 --1000
config.train_data.is_training = true

-- The trainer
config.train = {}
config.train.epoches = 150000
local baseRate = 0.05 / config.train_data.batch_size
local finetuneEpoches = 100
config.train.rates = {}
for i = 1, config.train.epoches / finetuneEpoches do
	config.train.rates[1 + (i - 1) * finetuneEpoches] = baseRate / i
end

config.train.momentum = 0.9
config.train.decay = 0.02
config.train.clamp = 10
-- config.train.recapture = true

-- Main program
config.main = {}
-- config.main.type = "torch.CudaTensor"
config.main.type = "torch.FloatTensor"
config.main.save = paths.concat(paths.cwd(), '../output/')
config.main.src_vocab_file = paths.concat(paths.cwd(), '../vocab.en.txt')
config.main.dst_vocab_file = paths.concat(paths.cwd(), '../vocab.ch.txt')
config.main.details = false
config.main.device = 1
config.main.logtime = 5
config.main.debug = false
config.main.test = true
