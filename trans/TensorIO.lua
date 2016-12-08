
local TensorIO = torch.class("TensorIO")

function TensorIO.loadTensor(filename, tensor)
   local file = io.open(filename, 'r')
   local tensor = TensorIO.readTensor(file, tensor)
   file:close()
end

function TensorIO.readTensor(file, tensor)

   function parseSizeLine(line)
      line = line:sub(line:find('of size ') + string.len('of size '), line:find('[,%]]') - 1)
      line = line:split('x')
      local size = torch.LongStorage(#line)
      for i = 1, #line do size[i] = tonumber(line[i]) end
      return size
   end

   function checkSizeLine()
      local line = file:read()
      line = line:gsub('^%s*(.-)%s*$', '%1') -- trim
      local size = parseSizeLine(line)
      local size2 = tensor:size()
      if #size ~= #size2 then return false end
      for i = 1, #size do
         if size[i] ~= size2[i] then return false end
      end
      return true
   end

   function fillTensor(tensor)
      if tensor:dim() > 1 then
         for i = 1, tensor:size(1) do
            fillTensor(tensor:select(1, i))
         end
      else
         while true do
            local line = file:read()
            line = line:gsub('^%s*(.-)%s*$', '%1') -- trim
            if line:len() ~= 0 and line:sub(1,1) ~= '(' then
               local data = line:split('%s+')
               for i = 1, #data do
                  tensor[i] = tonumber(data[i])
               end
               break
            end
         end
      end
   end

   if tensor:dim() == 1 then
      local len = tensor:size(1)
      fillTensor(tensor:view(len, 1))
      checkSizeLine()
   else
      fillTensor(tensor)
      checkSizeLine()
   end

end

function TensorIO.saveTensor(tensor, filename)
   local file = io.open(filename, 'w')
   TensorIO.writeTensor(tensor, file)
   file:close()
end

function TensorIO.writeTensor(tensor, file)

   local function _writeToFile12(tensor)
      if tensor:dim() == 1 then
         for i = 1, tensor:size(1) do
            file:write((tensor[i] == 0 and '0' or string.format('%0.4f', tensor[i])) .. '\n')
         end
      elseif tensor:dim() == 2 then
         for i = 1, tensor:size(1) do
            for j = 1, tensor:size(2) do
               file:write((tensor[i][j] == 0 and '0' or string.format('%0.4f', tensor[i][j])) .. '\t')
            end
            file:write('\n')
         end
      end
   end

   local function _writeToFile3plus(tensor, indexes)
      if tensor:dim() == 2 then
         local text = '('
         for ii, i in ipairs(indexes) do text = text .. i .. ',' end
         file:write(text .. '.,.) = \n')
         _writeToFile12(tensor)
      else
         for i = 1, tensor:size(1) do
            table.insert(indexes, i)
            _writeToFile3plus(tensor:select(1, i), indexes)
            table.remove(indexes, #indexes)
         end
      end
   end

   local function _writeToFile(tensor)
      if tensor:dim() == 1 or tensor:dim() == 2 then
         _writeToFile12(tensor)
      else    
         _writeToFile3plus(tensor, {})
      end
      file:write('[Tensor of size')
      local size = tensor:size()
      for i = 1, #size do file:write((i == 1 and ' ' or 'x') .. size[i]) end
      file:write(', storage ' .. #tensor:storage() .. '(' .. tensor:storageOffset() .. ')]\n')
   end

   _writeToFile(tensor)
end

--[[
local a = torch.rand(4,5,6)
--print(a)
TensorIO.saveTensor(a, "1.txt")
b = a.new():resizeAs(a)
TensorIO.loadTensor("1.txt", b)
print(b)

--[[]]--

return TensorIO