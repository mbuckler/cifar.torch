require 'torch'

--os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
--os.execute('tar -xvf cifar-10-binary.tar.gz')
local function convertCifar10BinToTorchTensor(inputFnames, outputFname)
   local nSamples = 0
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
      nSamples = nSamples + nSamplesF
      m:close()
   end

   local labels = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)

   local index = 1
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      m:seek(1)
      for j=1,nSamplesF do
         labels[index] = m:readByte()
         local store = m:readByte(3072)
         data[index]:copy(torch.ByteTensor(store))
         index = index + 1
      end
      m:close()
   end

   -- Add 1 to labels so as to be compatible with CrossEntropyCriterion
   labels:add(1)

   local out = {}
   out.data = data
   out.labels = labels
   print(out)
   torch.save(outputFname, out)
end


local cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Options:')
cmd:option('-path','/datasets/cifar-10/v0/','path to data to be converted')
cmd:text()

params = cmd:parse(arg)

cifar_path = params.path

print("Converting at path: ",cifar_path)
print("Set path with -path argument if desired")

convertCifar10BinToTorchTensor({
        (cifar_path .. 'data_batch_1.bin'),
        (cifar_path .. 'data_batch_2.bin'),
        (cifar_path .. 'data_batch_3.bin'),
        (cifar_path .. 'data_batch_4.bin'),
        (cifar_path .. 'data_batch_5.bin')},
                (cifar_path .. 'cifar10-train.t7'))

convertCifar10BinToTorchTensor({(cifar_path .. 'test_batch.bin')},
                (cifar_path .. 'cifar10-test.t7'))
                                                                                    73,1          Bot


