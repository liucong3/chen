require 'Dynamic'
require 'GridGRU'
require 'Lookup'

local Model, parent = torch.class('nn.Model', 'nn.Dynamic')

--[[
config.src_vocab
config.dst_vocab
config.src_emb
config.dst_emb
config.input_size
config.output_size
config.is_training
]]--
function Model:__init(config)
	self._type = torch.getdefaulttensortype()
	self.config = config
	self.srcLookup = nn.Lookup(config.src_vocab, config.src_emb)
	self.dstLookup = nn.Lookup(config.dst_vocab, config.dst_emb)
	self.forwRNN = nn.GridGRU({config.src_emb, config.dst_emb})
	self.backRNN = nn.GridGRU({config.src_emb, config.dst_emb})
	--self.dstClass = nn.Linear(config.dst_emb, config.dst_vocab, false)
	parent.__init(self, self.srcLookup, self.dstLookup, self.forwRNN, self.backRNN)
	-- srcLookup shares parameter with srcClass
	self.dstClass = nn.Linear(1, 1, false) -- no bias
	self.dstClass.weight = self.dstLookup.weight
	self.dstClass.gradWeight = self.dstLookup.gradWeight
	self:options(self.dstClass)
end

function Model:updateOutput(input)
	local groundTruth = nil
	if self.config.is_training then
		self:setInput(unpack(input))
		groundTruth = input[2]
		input = input[1]
	else
		self:setInput(input)
	end
	local input_size = self.config.input_size
	local output_size = self.config.output_size
	local batchSize = input:size(2)

	-- prepare word vector input
	local srcHidden = {}
	for i = 1, input_size do
		srcHidden[i] = self:DF(self.srcLookup, input:select(1, i))
	end

	-- predict output
	local output = input.new():resize(output_size, batchSize):fill(1)
	local dstHidden = self:DF(self.dstLookup, torch.LongTensor(batchSize):fill(1))
	if self.config.is_training then
		output = {}
	end
	for o = 2, output_size do
		for i = 1, input_size do
			srcHidden[i], dstHidden = self:DF(self.forwRNN, srcHidden[i], dstHidden)
		end
		for i = input_size, 1, -1 do
			srcHidden[i], dstHidden = self:DF(self.backRNN, srcHidden[i], dstHidden)
		end
		local predict = self:DF(self.dstClass, dstHidden)
		--predict = self:DF(nn.LogSoftMax(), predict)

		if self.config.is_training then
			output[o - 1] = predict
			dstHidden = self:DF(self.dstLookup, groundTruth:select(1, o))
		else
			local _, index = predict:max(2)
			index = index:view(-1)
			output:select(1, o):copy(index)
			dstHidden = self:DF(self.dstLookup, index)
		end
	end

	if self.config.is_training then
		return self:setOutput(unpack(output))
	else
		return self:setOutput(output)
	end
end

--[[
require 'gradientCheck'
local config = {}
config.src_vocab = 10
config.dst_vocab = 10
config.src_emb = 7
config.dst_emb = 7
config.input_size = 5
config.output_size = 10
config.is_training = true
local model = nn.Model(config)
local param = model:getParameters()
param:copy(torch.randn(param:size()) * 0.01)
local input = torch.LongTensor{{1,1},{2,3},{4,5},{6,7},{8,9}}
local truth = torch.LongTensor{{1,1},{2,3},{4,5},{6,7},{8,9},{2,4},{2,3},{4,5},{6,7},{8,9}}
local output = model:forward({input, truth})
--print(output)

--print(model.dstClass.weight)
gradientCheck(model, {input, truth})
--[[]]--

require 'gradientCheck'
local config = {}
config.src_vocab = 6
config.dst_vocab = 5
config.src_emb = 4
config.dst_emb = 3
config.input_size = 3
config.output_size = 5
config.batch_size = 3
config.is_training = true
local model = nn.Model(config)
local param = model:getParameters()
param:copy(torch.randn(param:size()) * 0.01)
local batch_size = 3
local input = torch.LongTensor(config.input_size, config.batch_size)
local truth = torch.LongTensor(config.output_size, config.batch_size)
input:copy(torch.rand(config.input_size, config.batch_size) * config.src_vocab * 1.5 + 1)
truth:copy(torch.rand(config.output_size, config.batch_size) * config.src_vocab * 1.5 + 1)
local output = model:forward({input, truth})
--print(output)

--print(model.dst_embClass.weight)
gradientCheck(model, {input, truth})
