classdef TestTensorStack < matlab.unittest.TestCase
   
    properties
        stack       % tested TensorStack
        real_stack  % Corresponding regular array
    end
    
    properties (TestParameter)
        % test indexing
        subs = struct( ...
            'all', {{':', ':', ':'}}, ...
            'flat', {{':'}}, ...
            'wrapped', {{':', ':'}}, ...
            'zero', {{2:1, ':', ':'}}, ...
            'zero2', {{':', 2:1, ':'}}, ...
            'zero3', {{':', ':', 2:1}}, ...
            'part', {{':', ':', 3}}, ...
            'part2', {{3, ':', ':'}}, ...
            'part3', {{1:3, ':', ':'}}, ...
            'part4', {{1:3, ':', 4:5}}, ...
            'part5', {{1:3, 3, 4:5}}, ...
            'part6', {{':', 3:end, 4:5}}, ...
            'part7', {{':', [3, 1, 2], ':'}});
        % test permutations
        order = struct( ...
            'unchanged', [1, 2, 3], ...
            'reverse', [3, 2, 1], ...
            'invert1', [2, 1, 3], ...
            'invert2', [1, 3, 2], ...
            'shuffle', [2, 3, 1]);
        % test reshaping (split only)
        newsize = struct( ...
            'unchanged', [3, 12, 5], ...
            'unknown1', {{3, [], 5}}, ...
            'unknown2', {{3, [], 6, 5}}, ...
            'unknown3', {{[], 2, 6, 5}}, ...
            'dummy1', [3, 12, 5, 1], ...
            'dummy2', [3, 12, 1, 5], ...
            'dummy3', [3, 1, 1, 1, 12 5], ...
            'split1', [3, 2, 6, 5], ...
            'split2', [3, 3, 4, 5], ...
            'split3', [3, 4, 3, 5], ...
            'split4', [3, 2, 2, 3, 5], ...
            'split5', [3, 3, 2, 2, 5]);
        % seed for random permutations
        permseed = {1, 22, 542, 445, 55324};
    end

    methods (TestMethodSetup)
        function createStack(testCase)
            rng(1)
            data1 = randn(3, 4, 5);
            data2 = randn(3, 7, 5);
            data3 = randn(3, 1, 5);
            testCase.stack = TensorStack(2, data1, data2, data3);
            testCase.real_stack = cat(2, data1, data2, data3);
        end
    end

    methods (Test)
        function testSubsref(testCase, subs)
            testCase.verifyEqual( ...
                testCase.stack(subs{:}), testCase.real_stack(subs{:}));
        end

        function testPermute(testCase, order)
            pstack = permute(testCase.stack, order);
            preal = permute(testCase.real_stack, order);
            testCase.verifySize(pstack, size(preal));
            testCase.verifyEqual(pstack(:, :, :), preal(:, :, :));
        end
        
        function testReshape(testCase, newsize)
            if iscell(newsize)
                pstack = reshape(testCase.stack, newsize{:});
                preal = reshape(testCase.real_stack, newsize{:});
            else
                pstack = reshape(testCase.stack, newsize);
                preal = reshape(testCase.real_stack, newsize);
            end
            testCase.verifySize(pstack, size(preal));
            testCase.verifyEqual(pstack(:, :, :), preal(:, :, :));
        end

        function testReshapePermute(testCase, newsize, permseed)
            if iscell(newsize)
                pstack = reshape(testCase.stack, newsize{:});
                preal = reshape(testCase.real_stack, newsize{:});
            else
                pstack = reshape(testCase.stack, newsize);
                preal = reshape(testCase.real_stack, newsize);
            end

            rng(permseed)
            rorder = randperm(ndims(preal));
            pstack = permute(pstack, rorder);
            preal = permute(preal, rorder);

            testCase.verifySize(pstack, size(preal));
            testCase.verifyEqual(pstack(:, :, :), preal(:, :, :));
        end

    end
end