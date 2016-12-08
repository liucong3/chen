require 'nn'
require 'torch'

--[[
This class allow out-of-range index to be ignored.
]]--

local Lookup, parent = torch.class('nn.Lookup', 'nn.Module')

function Lookup:__init(vocabSize, embeddingSize, unk, weight, gradWeight)
	self.unk = unk or vocabSize
	self.vocabSize = vocabSize
	self.output = torch.Tensor()
	--self.gradInput = torch.Tensor()
	self.weight = weight or torch.Tensor(vocabSize, embeddingSize)
	self.gradWeight = gradWeight or torch.Tensor(vocabSize, embeddingSize)
end

function Lookup:updateOutput(input)
	self._input = input
	local out = torch.gt(input, self.vocabSize)
	if out:sum() > 0 then
		out = out:typeAs(input)
		self._input = input:clone()
		self._input:add(torch.mul(out, self.unk) - torch.cmul(out, input))
		--print(input, out, self._input)
	end
	self.output:index(self.weight, 1, self._input)
	return self.output
end

function Lookup:updateGradInput(input, gradOutput)
	return self.gradInput
end

function Lookup:accGradParameters(input, gradOutput, scale)
	if scale and scale ~= 1 then gradOutput = torch.mul(gradOutput, scale) end
	--self.gradWeight:indexAdd(1, self._input, gradOutput)
	for i = 1, self._input:size(1) do
		self.gradWeight:select(1, self._input[i]):add(gradOutput:select(1, i))
	end
end

function Lookup:shareClone()
	local size = self.weight:size()
	return nn.Lookup(size[1], size[2], self.unk, self.weight, self.gradWeight)
end

--[[
require 'Check'
require 'MSELoss'
local c = Check()
local lookup = nn.Lookup(10,5)
c.randParams(lookup, 0.1)
local input = c.randIndex({{1,7}}, {10})[1]:view(-1)
local label = c.randData({{7,5}})[1]
print(input, label)
c.gradCheck(lookup, nn.MSELoss(), input, label, 1e-6)
--[[]]--

--[[
require 'gradientCheck'
lookup = nn.Lookup(10,5)
lookup.weight:fill(0)
lookup.weight:narrow(2, 1, 1):copy(torch.range(1, 10))
--print(lookup.weight)
input = torch.LongTensor({1,1,7})
output = lookup:forward(input)
--print(lookup:forward(input))
gradOutput = torch.Tensor(output:size()):fill(1)
gradInput = lookup:backward(input, gradOutput)
lookup:updateParameters(0.1)
gradientCheck(lookup, input)
--[[]]--

--[[
require 'gradientCheck'
seq = nn.Sequential()
seq:add(nn.Lookup(10,5))
seq:add(nn.Linear(5,3))
input = torch.LongTensor({1,1,7})
gradientCheck(seq, input)
--[[]]--


--[[
require 'gradientCheck'
require 'Dynamic'

local Test, parent = torch.class('nn.Test', 'nn.Dynamic')

function Test:__init(lookup, linear)
    self.lookup = lookup or nn.Lookup(4,3)
    --self.lookup = lookup or nn.Linear(4,3)
    self.linear = linear or nn.Linear(3,2)
    parent.__init(self, self.lookup, self.linear)
    --parent.__init(self, false, false, self.lookup)
end

function Test:updateOutput(input)
	self.forwSequence = {}
	local hidden = self:DF(self.lookup, input)
	--return self:setOutput(hidden)
	local output = self:DF(self.linear, hidden)
	return self:setOutput(output)
end

function Test:shareClone()
	return nn.Test(self.lookup, self.linear)
end

--local lookup = nn.Lookup(4,3)
--lookup.weight:select(2,1):copy(torch.range(1,4))
--local test = nn.Test(lookup)
local test = nn.Test()
--print(test.lookup.weight)
--print(test:parameters())
local input = torch.LongTensor({1,4,3})
--local output = test:forward(input)
--print(output)
--local gradOutput = output:clone():fill(1)
--test:zeroGradParameters()
--test:updateGradInput(input, gradOutput)
--test:accGradParameters(input, gradOutput, 1)
--print('-->', test.lookup.gradWeight)
--print(test.lookup.gradWeight)
--local input = torch.randn(7,4)
gradientCheck(test, input)
--[[]]--

