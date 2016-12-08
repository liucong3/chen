require 'torch'
require 'nn'
require 'fixes'
-- require 'clnn'
-- require 'cltorch'
-- require 'clfixes'

require 'Data'
require 'Model1'
require 'Loss'
require 'Train'
require 'config'

if cltorch then
   torch.setdefaulttensortype('torch.ClTensor')
end

local function save_model(folder, config, param)
   --os.execute('mkdir ' .. folder)
   torch.save(folder .. '/config.t7', config)
   local tio = require 'TensorIO'
   tio.saveTensor(param:float(), folder .. '/best-param.txt')
end

local function load_model(folder, params)
   local config = torch.load(folder .. '/config.t7')
   local tio = require 'TensorIO'
   tio.loadTensor(folder .. '/best-param.txt', params)
   return config
end

print(os.date() .. ' -- Loading data ...')
local data = io.Data(config.train_data)
collectgarbage()
local model = nn.Model(config.train_data)
local loss = Loss()
local train = Train(data, model, loss, config.train)

-- print(os.date() .. ' - Loading model params ...')
-- model.config = load_model(config.main.save, train.params)

collectgarbage()
print(os.date() .. ' -- Training ...')

local min_loss = nil
train:run(config.train.epoches, function (_, epoch)
   local time = string.format('%.2f/%.2f/%.2f/%.2f', train.time.data, train.time.forward, train.time.backward, train.time.update)
   train:clearTime()
   local loss = train.objective
   min_loss = min_loss or loss
   local lossInfo = string.format('Loss=%.2f%s', loss, (min_loss > loss and '*' or ''))
   print('' .. epoch .. '. ' .. os.date() .. ' - ' .. time .. ' - ' .. lossInfo)
   if min_loss > loss then
      min_loss = loss
      save_model(config.main.save, model.config, train.params)
   end
   collectgarbage()
end)

