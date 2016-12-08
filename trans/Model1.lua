-- require 'torch'
-- require 'nn'
-- require 'fixes'
-- require 'clnn'
-- require 'cltorch'
-- require 'clfixes'
-- torch.setdefaulttensortype('torch.ClTensor')

require 'Dynamic'
require 'GridLSTM'
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
	self.dstLookup = nn.Lookup(config.dst_vocab, 2 * config.dst_emb)
	self.forwRNN = nn.GridLSTM({config.src_emb, config.dst_emb})
	self.backRNN = nn.GridLSTM({config.src_emb, config.dst_emb})
	--self.batchNorm = nn.BatchNormalization(config.dst_emb)
	--self.dstClass = nn.Linear(config.dst_emb, config.dst_vocab, false)
	parent.__init(self, self.srcLookup, self.dstLookup, self.forwRNN, self.backRNN)
	-- srcLookup shares parameter with srcClass
	self.dstClass = nn.Linear(1, 1, false) -- no bias
	self.dstClass.weight = self.dstLookup.weight
	self.dstClass.gradWeight = self.dstLookup.gradWeight
	self:options(self.dstClass)
	--[[]]--
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
	local srcCell = {}
	local srcHidden = {}
	for i = 1, input_size do
		local embSrc = self:DF(self.srcLookup, input:select(1, i))
		srcCell[i], srcHidden[i] = embSrc, embSrc
	end

	-- predict output
	local output = input.new():resize(output_size, batchSize):fill(1)
	local embDst = self:DF(self.dstLookup, torch.LongTensor(batchSize):fill(1))
	if self.config.is_training then
		output = {}
	end
	for o = 2, output_size do
		local dstCell, dstHidden = self:DF(nn.BranchTable{self.config.dst_emb, self.config.dst_emb}, embDst)
		for i = 1, input_size do
			srcCell[i], dstCell, srcHidden[i], dstHidden = self:DF(self.forwRNN, srcCell[i], dstCell, srcHidden[i], dstHidden)
		end
		for i = input_size, 1, -1 do
			srcCell[i], dstCell, srcHidden[i], dstHidden = self:DF(self.backRNN, srcCell[i], dstCell, srcHidden[i], dstHidden)
		end
		embDst = self:DF(nn.MergeTable(2), dstCell, dstHidden)
		local predict = self:DF(self.dstClass, embDst)

		if self.config.is_training then
			output[o - 1] = predict
			embDst = self:DF(self.dstLookup, groundTruth:select(1, o))
		else
			local _, index = predict:max(2)
			index = index:view(-1)
			output:select(1, o):copy(index)
			embDst = self:DF(self.dstLookup, index)
		end
	end

	if self.config.is_training then
		return self:setOutput(unpack(output))
	else
		return self:setOutput(output)
	end
	--[[]]--
end

--[[

require 'Check'
require 'Loss'
local c = Check()
local config = {}
config.src_vocab = 2
config.dst_vocab = 2
config.src_emb = 2
config.dst_emb = 2
config.input_size = 2
config.output_size = 3
config.batch_size = 2
config.is_training = true
local model = nn.Model(config)
c.randParams(model)
local param = model:getParameters()
param:copy(torch.randn(param:size()) * 0.1)
local input = c.randIndex({{config.input_size, config.batch_size}, 
	{config.output_size, config.batch_size}},
	{config.src_vocab, config.dst_vocab})
local label = c.randIndex({{config.output_size, config.batch_size}},
	{config.dst_vocab})

if cltorch then
	for i = 1, #input do
		input[i] = input[i]:type(torch.getdefaulttensortype())
	end
	for i = 1, #label do
		label[i] = label[i]:type(torch.getdefaulttensortype())
	end
end

--label = c.randData({{config.batch_size, config.src_emb}, {config.batch_size, config.src_emb}})

--require 'MSELoss'
c.gradCheck(model, Loss(), input, label[1], 1e-6)
--[[]]--

--[[
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

--print(model.dstClass.weight)
gradientCheck(model, {input, truth})
--[[]]--

