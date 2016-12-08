require 'io'
require 'torch'

local Data = torch.class('io.Data')

function Data:__init(config)
   self.batch_size = config.batch_size
   self.input_size = config.input_size
   self.output_size = config.output_size
   self.data = Data._loadCorpus(config.file)
end

function Data._loadCorpus(filename)
	local data = {}
	local file = io.open(filename, 'r')
	local count = file:read('*n')
	--count = 1000 -- TOREMOVE
	for i = 1, count do
		local en_count = file:read('*n')
		local en = torch.LongTensor(en_count)
		for j = 1, en_count do
			en[j] = file:read('*n')
		end
		local ch_count = file:read('*n')
		local ch = torch.LongTensor(ch_count)
		for j = 1, ch_count do
			ch[j] = file:read('*n')
		end
		data[i] = { en, ch }
	end
	file:close()
	return data
end

function Data:getBatch(inputs, outputs, data)
	local data = data or self.data
	local inputs = inputs or torch.LongTensor(self.batch_size, self.input_size)
	local outputs = outputs or torch.LongTensor(self.batch_size, self.output_size)
	inputs:select(2, 1):fill(1) -- <s>
	inputs:narrow(2, 2, self.input_size-1):fill(2) -- </s>
	outputs:select(2, 1):fill(1) -- <s>
	outputs:narrow(2, 2, self.output_size-1):fill(2) -- </s>

	for i = 1, inputs:size(1) do
		-- Choose data
		local index = torch.IntTensor.random(#data)
		local input_size = data[index][1]:size()[1]
		if input_size > self.input_size - 1 then input_size = self.input_size - 1 end
		inputs:select(1, i):narrow(1, 2, input_size):copy(data[index][1]:narrow(1, 1, input_size) + 2)

		local output_size = data[index][2]:size()[1]
		if output_size > self.output_size - 1 then output_size = self.output_size - 1 end
		outputs:select(1, i):narrow(1, 2, output_size):copy(data[index][2]:narrow(1, 1, output_size) + 2)
	end

	if cltorch then
		inputs = inputs:cl()
		outputs = outputs:cl()
	end
	return inputs, outputs
end

--[[
local data = io.Data{batch_size = 10, input_size = 30, output_size = 30, file = '../chen.code.txt'}
local inputs, outputs = nil, nil
for i = 1, 3 do	
	inputs, outputs = data:getBatch(inputs, outputs)
	print(inputs)
	print(outputs)
end
--[[]]--
