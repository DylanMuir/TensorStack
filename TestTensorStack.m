% Test suite for TensorStack
% Use 'run(TestTensorStack)' to run the whole suite.
classdef TestTensorStack < matlab.unittest.TestCase
   
    properties
        stack       % tested TensorStack
        real_stack  % corresponding regular array
    end
    
    properties (TestParameter)
        % tested indexing schemas
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
        % tested permutations
        order = struct( ...
            'unchanged', [1, 2, 3], ...
            'reverse', [3, 2, 1], ...
            'invert1', [2, 1, 3], ...
            'invert2', [1, 3, 2], ...
            'shuffle', [2, 3, 1]);
        % tested new size for reshaping (split only)
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
        % tested splits for 12
        splits = { ...
            [2, 6], [6, 2], [3, 4], [4, 3], [2, 2, 3], [3, 2, 2], [2, 3, 2]};
    end

    methods (TestMethodSetup)
        function createStack(testCase)
            % tested data, as a concatenation of 3 arrays
            rng(1)
            data1 = randn(3, 4, 5);
            data2 = randn(3, 7, 5);
            data3 = randn(3, 1, 5);
            testCase.stack = TensorStack(2, data1, data2, data3);
            testCase.real_stack = cat(2, data1, data2, data3);
        end
    end

    methods (Test)
        % test indexing
        function testSubsref(testCase, subs)
            testCase.verifyEqual( ...
                testCase.stack(subs{:}), testCase.real_stack(subs{:}));
        end

        % test permutation
        function testPermute(testCase, order)
            % permute stack
            pstack = permute(testCase.stack, order);
            preal = permute(testCase.real_stack, order);
            testCase.verifySize(pstack, size(preal));
            testCase.verifyEqual(pstack(:, :, :), preal);

            % restore original order
            ppstack = ipermute(pstack, order);
            testCase.verifySize(ppstack, size(testCase.real_stack));
            testCase.verifyEqual(ppstack(:, :, :), testCase.real_stack);
        end
        
        % test reshaping
        function testReshape(testCase, newsize)
            if iscell(newsize)
                rstack = reshape(testCase.stack, newsize{:});
                rreal = reshape(testCase.real_stack, newsize{:});
            else
                rstack = reshape(testCase.stack, newsize);
                rreal = reshape(testCase.real_stack, newsize);
            end
            colons = repmat({':'}, ndims(rreal), 1);
            testCase.verifySize(rstack, size(rreal));
            testCase.verifyEqual(rstack(colons{:}), rreal);
        end

        % test reshaping + random permutation
        function testReshapePermute(testCase, newsize, permseed)
            if iscell(newsize)
                rstack = reshape(testCase.stack, newsize{:});
                rreal = reshape(testCase.real_stack, newsize{:});
            else
                rstack = reshape(testCase.stack, newsize);
                rreal = reshape(testCase.real_stack, newsize);
            end

            % random permutation
            rng(permseed)
            rorder = randperm(ndims(rreal));
            pstack = permute(rstack, rorder);
            preal = permute(rreal, rorder);

            colons = repmat({':'}, ndims(preal), 1);
            testCase.verifySize(pstack, size(preal));
            testCase.verifyEqual(pstack(colons{:}), preal);

            % restore shape before permutation
            ppstack = ipermute(pstack, rorder);
            colons = repmat({':'}, ndims(rreal), 1);
            testCase.verifySize(ppstack, size(rreal));
            testCase.verifyEqual(ppstack(colons{:}), rreal);
        end
        % test permutation + reshape
        function testPermuteReshape(testCase, order, splits)
            % permutation
            pstack = permute(testCase.stack, order);
            preal = permute(testCase.real_stack, order);

            % split dimension who is equal to 12
            psize = size(preal);
            split_dim = find(psize == 12);
            rsize = [psize(1:(split_dim-1)), splits, psize((split_dim+1:end))];

            % reshape and test
            rstack = reshape(pstack, rsize);
            rreal = reshape(preal, rsize);
            colons = repmat({':'}, ndims(rreal), 1);
            testCase.verifySize(rstack, size(rreal));
            testCase.verifyEqual(rstack(colons{:}), rreal);
        end
    end
end