require 'torch'
require 'nn'
require 'fixes'
require 'clnn'
require 'cltorch'
require 'GridLSTM'

function torch.ClTensor:uniform(a, b)
   if a == nil then
      a = 0
   end
   if b == nil then
      b = 1
   end
   self:copy(torch.FloatTensor(self:size()):uniform(a, b))
   return self
end

local randn0 = torch.randn
local rand0 = torch.rand

torch.randn = function(...)
  print('torch.randn')
  local tensorType = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local tensor = randn0(...)
  torch.setdefaulttensortype(tensorType)
  return tensor
end

torch.rand = function(...)
  print('torch.rand')
  local tensorType = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local tensor = rand0(...)
  torch.setdefaulttensortype(tensorType)
  return tensor
end

-------------------

torch.setdefaulttensortype('torch.ClTensor')

local input_sizes = {3, 4, 5}
local lstm = nn.GridLSTM(input_sizes)
local input_sizes2 = { unpack(input_sizes) }
for i = 1, #input_sizes do
    table.insert(input_sizes2, input_sizes[i])
end
--gradientCheck(lstm, 10, input_sizes2)


-------------------

print('Test speed')

local batchSize = 1000
input_sizes = {1000, 1000, 1000}
input_sizes2 = { unpack(input_sizes) }
for i = 1, #input_sizes do
    table.insert(input_sizes2, input_sizes[i])
end

function createInput(batchSize, input_sizes)
    input = {}
    for i = 1, #input_sizes do
        input[i] = torch.Tensor(batchSize, input_sizes[i]):fill(1)
    end
    return input
end

print('torch.FloatTensor')
torch.setdefaulttensortype('torch.FloatTensor')
lstm = nn.GridLSTM(input_sizes)
input = createInput(batchSize, input_sizes2)
for i = 1, 10 do
    print(i)
    input = lstm:forward(input)
end

print('torch.ClTensor')
torch.setdefaulttensortype('torch.ClTensor')
lstm = nn.GridLSTM(input_sizes)
input = createInput(batchSize, input_sizes2)
for i = 1, 100 do
    print(i)
    input = lstm:forward(input)
end


