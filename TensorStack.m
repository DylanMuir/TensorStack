% TensorStack - CLASS Create a seamless stack of tensors
%
% Usage: tsTensor = TensorStack(nCatDim, tfTensor1 <, tfTensor2, tfTensor3, ...>)
%
% 'nCatDim' is a dimension along which to contacentate several tensors.
% 'tfTensor1', ..., 'tfTensorN' must all have identical sizes, except for
% along the dimensions 'nCatDim'.
%
% 'tsTensor' will be a TensorStack object, which appears like a normal
% matlab tensor. It can be referenced using normal matlab indexing, with
% some restrictions. All-colon referencing is unrestricted. However, when
% any subscript contains a non-colon reference, then the dimensions up
% until and including the concatenated stack dimension 'nCatDim' must be
% referenced individually. Subsequent dimensions may be wrapped up using
% partial linear referencing.
%
% For example:
%
% >> tsTensor = TensorStack(1, ones(3, 3, 3), ones(3, 3, 3))
%
% tsTensor = 
%    6x3x3 TensorStack object.
%
% >> size(tsTensor(:))
%
% ans = [54 1]
%
% >> size(tsTensor(:, :))
%
% ans = [3 18]
%
% >> tsTensor(12)
%
% Error using TensorStack/my_subsref (line 178)
% *** TensorStack: Only limited referencing styles are supported. The concatenated stack dimension [2] must be referenced independently.
%
% >> size(tsTensor(2, :))
%
% ans = [1 9]
%
% >> tsTensor(1, 7)
%
% ans = 1

% Author: Dylan Muir <dylan.muir@unibas.ch>
% Created: July 2015

classdef TensorStack
   properties (SetAccess = private, GetAccess = private)
      nStackDim;        % - Dimension along which concatenation has taken place
      ctTensors;        % - Cell array of sub-tensors
      cvnTensorSizes;   % - Cell array containing buffered sizes
      strDataClass;     % - Buffered data class
      vnDimsOrder;      % - Internal dimensions order to support permutation
   end
   
   methods
      %% TensorStack - Constructor
      function oStack = TensorStack(varargin)
         % -- Check arguments
         if (nargin < 2)
            help TensorStack;
            error('TensorStack:Usage', ...
                  '*** TensorStack: Incorrect usage.');
         end
         
         % - Buffer tensor sizes
         oStack.nStackDim = varargin{1};
         oStack.cvnTensorSizes = cellfun(@size, varargin(2:end), 'UniformOutput', false);

         % - Check dimension argument
         if (~isnumeric(oStack.nStackDim) || (oStack.nStackDim < 1))
            error('TensorStack:Arguments', ...
                  '*** TensorStack: Stacking dimension must be a number > 1.');
         end
         
         % - Check that all tensors have the same size (apart from stack dimension)
         cvnCheckDims = oStack.cvnTensorSizes;
         for nTensor = 1:numel(cvnCheckDims)
            cvnCheckDims{nTensor}(oStack.nStackDim) = 0;
         end
         
         if (~all(cellfun(@(c)isequal(c, cvnCheckDims{1}), cvnCheckDims(2:end))))
            error('TensorStack:Arguments', ...
                  '*** TensorStack: All tensors must be of the same dimensions, apart from stacking dimension.');
         end
         
         % - Check that all tensors have the same data class
         oStack.strDataClass = class(varargin{2}(1));
         if (~all(cellfun(@(c)isequal(class(c(1)), oStack.strDataClass), varargin(2:end))))
            error('TensorStack:Arguments', ...
                  '*** TensorStack: All tensors must be of the same numeric class.');
         end

         % - Initialize dimensions order
         oStack.vnDimsOrder = 1:numel(cvnCheckDims{1});

         % -- Store sub-tensors
         oStack.ctTensors = varargin(2:end);
      end
      
      
      %% -- Overloaded subsref, subsasgn
      function [varargout] = subsref(oStack, S)
         % - More than one return argument means cell or dot referencing was
         % used
         if (nargout > 1)
            error('TensorStack:InvalidReferencing', ...
               '*** TensorStack: ''{}'' and ''.'' referencing methods are not supported by TensorStack objects.');
         end
         
         % - Check reference type
         switch (S(1).type)
            case {'.', '{}'}
               % - Unsupported referencing
               error('TensorStack:InvalidReferencing', ...
               '*** TensorStack: ''{}'' and ''.'' referencing methods are not supported by TensorStack objects.');
               
            case {'()'}
               % - Call the internal subsref function
               [varargout{1}] = my_subsref(oStack, S);

            otherwise
               % - Unknown referencing type
               error('TensorStack:UnknownReferenceType', ...
                  '*** TensorStack: An unknown referencing method was used.');
         end
      end
      
      % subsasgn - Overloaded subsasgn
      function varargout = subsasgn(~, ~, ~) %#ok<STOUT>
         error('TensorStack:NotSupported', ...
               '*** TensorStack: Assignment is not supported.');
      end
      
      % other_subsref - Standard array referencing
      function [tfData] = my_subsref(oStack, S)
         % - Retrieving stack size information
         vnRefTensorSize = size(oStack);

         % - Cleaning input indices (colon or linear indices)
         coSubs = cleansubs(S.subs, vnRefTensorSize);
         nNumDims = numel(coSubs);
         vbIsColon = cellfun(@iscolon, coSubs);

         % - Output data size, taking into accound wrapped-up tensor dimensions
         vnWrappedTensorSize = vnRefTensorSize(1:nNumDims);
         vnWrappedTensorSize(nNumDims) = prod(vnRefTensorSize(nNumDims:end));
         vnDataSize = vnWrappedTensorSize;
         vnDataSize(~vbIsColon) = cellfun(@nnz, coSubs(~vbIsColon));

         % - Catch "all colon" entire stack referencing
         if all(vbIsColon)
            tfData = oStack.retrieve_all();
            tfData = reshape(tfData, vnDataSize);
            return
         end

         % - Catch empty reference
         if (prod(vnDataSize) == 0)
            tfData = zeros(vnDataSize, oStack.strDataClass);
            return;
         end

         % - Forbid wrapped-up dimensions for permuted dimensions
         %TODO

         % - Forbid wrapped-up dimensions before the concatenation dimension
         if (nNumDims < numel(vnRefTensorSize)) && (nNumDims <= oStack.nStackDim)
            error('TensorStack:badsubscript', ...
                  '*** TensorStack: Only limited referencing styles are supported. The concatenated stack dimension [%d] must be referenced independently.', ...
                  oStack.nStackDim);
         end

         % - Check stack references
         if any(cellfun(@max, coSubs(~vbIsColon)) > vnWrappedTensorSize(~vbIsColon))
            error('TensorStack:badsubscript', ...
                  'Index exceeds matrix dimensions.');
         end

         % - Permute data size and indices
         vnInvOrder(oStack.vnDimsOrder(1:nNumDims)) = 1:nNumDims;
         vnOrigSize = vnDataSize(vnInvOrder);
         cOrigSubs = coSubs(vnInvOrder);

         % - Retrieve data from sub-tensors and permute it
         tfData = oStack.retrieve_part(cOrigSubs, vnOrigSize);
         tfData = permute(tfData, oStack.vnDimsOrder(1:nNumDims));
      end

      % retrieve_part - Retrieve part of the stack, using original size
      function [tfData] = retrieve_part(oStack, cSubs, vnDataSize)
         % - Allocate return tensor, using buffered data class
         tfData = zeros(vnDataSize, oStack.strDataClass);

         % - Loop over sub-tensors, referencing them in turn
         nStackStart = 1;
         nCurrentIdx = 1;

         for nSubTensor=1:numel(oStack.ctTensors)
            nNCatDim = oStack.cvnTensorSizes{nSubTensor}(oStack.nStackDim);
            nStackEnd = nStackStart + nNCatDim - 1;

            % - Check if the whole concatenated dimension is retrieved
            if iscolon(cSubs{oStack.nStackDim})
               cTensorSubs = cSubs;
               nElements = nNCatDim;

            else
               % - Work out which indices are within this tensor
               vbThisTensor = ...
                  (cSubs{oStack.nStackDim} >= nStackStart) & ...
                  (cSubs{oStack.nStackDim} <= nStackEnd);
               nElements = nnz(vbThisTensor);

               % - Map tensor subscripts
               cTensorSubs = cSubs;
               cTensorSubs{oStack.nStackDim} = ...
                  cSubs{oStack.nStackDim}(vbThisTensor) - nStackStart + 1;
            end

            if nElements > 0
               % - Map data subscripts
               cDataSubs = repmat({':'}, 1, numel(cSubs));
               cDataSubs{oStack.nStackDim} = ...
                   nCurrentIdx:(nCurrentIdx + nElements - 1);

               % - Access sub-tensor data
               tfData(cDataSubs{:}) = oStack.ctTensors{nSubTensor}(cTensorSubs{:});
            end

            nStackStart = nStackEnd + 1;
            nCurrentIdx = nCurrentIdx + nElements;
         end
      end

      % retrieve_all - Retrieve all data from the stack
      function [tfData] = retrieve_all(oStack)
         % - Allocate return tensor, using buffered data class
         tfData = zeros(oStack.insize(), oStack.strDataClass);

         % - Loop over sub-tensors, referencing them in turn
         nCurrentIdx = 1;
         cTensorSubs = repmat({':'}, 1, ndims(oStack));

         for nSubTensor=1:numel(oStack.ctTensors)
            nCatDim = oStack.cvnTensorSizes{nSubTensor}(oStack.nStackDim);

            % - Map data subscripts
            cDataSubs = cTensorSubs;
            cDataSubs{oStack.nStackDim} = ...
               nCurrentIdx:(nCurrentIdx + nCatDim - 1);

            % - Access sub-tensor data
            tfData(cDataSubs{:}) = oStack.ctTensors{nSubTensor}(cTensorSubs{:});

            nCurrentIdx = nCurrentIdx + nCatDim;
         end

         % - Permute dimensions
         tfData = permute(tfData, oStack.vnDimsOrder);
      end


      %% -- Overloaded size, numel, end, etc

      % insize - Un-manipulated internal size of the tensor
      function vnSize = insize(oStack)
         vnSize = oStack.cvnTensorSizes{1};
         vnStackLengths = cellfun(@(c)c(oStack.nStackDim), oStack.cvnTensorSizes);
         vnSize(oStack.nStackDim) = sum(vnStackLengths);
      end

      % size - METHOD Overloaded size
      function varargout = size(oStack, vnDimensions)
         % - Get tensor stack size and permute it
         vnSize = oStack.insize();
         vnSize = vnSize(oStack.vnDimsOrder);

         % - Return specific dimension(s)
         if (exist('vnDimensions', 'var'))
            if (~isnumeric(vnDimensions) || ~all(isreal(vnDimensions)))
               error('TensorStack:dimensionMustBePositiveInteger', ...
                  '*** TensorStack: Dimensions argument must be a positive integer within indexing range.');
            end
            
            % - Return the specified dimension(s)
            vnSize = vnSize(vnDimensions);
         end
         
         % - Handle differing number of size dimensions and number of output
         % arguments
         nNumArgout = max(1, nargout);
         
         if (nNumArgout == 1)
            % - Single return argument -- return entire size vector
            varargout{1} = vnSize;
            
         elseif (nNumArgout <= numel(vnSize))
            % - Several return arguments -- return single size vector elements,
            % with the remaining elements grouped in the last value
            varargout(1:nNumArgout-1) = num2cell(vnSize(1:nNumArgout-1));
            varargout{nNumArgout} = prod(vnSize(nNumArgout:end));
            
         else
            % - Output all size elements
            varargout(1:numel(vnSize)) = num2cell(vnSize);

            % - Deal out trailing dimensions as '1'
            varargout(numel(vnSize)+1:nNumArgout) = {1};
         end
      end
      
      % numel - METHOD Overloaded numel
      function [nNumElem] = numel(oStack, varargin)
         % - If varargin contains anything, a cell reference "{}" was attempted
         if (~isempty(varargin))
            error('TensorStack:cellRefFromNonCell', ...
               '*** TensorStack: Cell contents reference from non-cell obejct.');
         end
         
         % - Return the total number of elements in the tensor
         nNumElem = prod(size(oStack)); %#ok<PSIZE>
      end
      
      % ndims - METHOD Overloaded ndims
      function [nNumDims] = ndims(oStack)
         nNumDims = numel(size(oStack));
      end
      
      % end - METHOD Overloaded end
      function ind = end(obj,k,n)
         szd = size(obj);
         if k < n
            ind = szd(k);
         else
            ind = prod(szd(k:end));
         end
      end

      % permute - METHOD Overloaded permute
      function [oStack] = permute(oStack, vnNewOrder)
         % - Test permutation indices
         vnTestedOrder = sort(vnNewOrder(:));
         nNDims = numel(oStack.vnDimsOrder);

         try
            validateattributes(vnTestedOrder, {'numeric'}, ...
                {'positive', 'integer', 'numel', nNDims, 'increasing', '<=', nNDims});
         catch
            error('TensorStack:badorder', ...
                  '*** TensorStack: invalid permutation indices');
         end

         % - Change dimensions order
         oStack.vnDimsOrder(1:numel(vnNewOrder)) = oStack.vnDimsOrder(vnNewOrder(:));
      end

      % ipermute - METHOD Overloaded ipermute function
      function [oStack] = ipermute(oStack, vnOldOrder)
         vnNewOrder(vnOldOrder) = 1:numel(vnOldOrder);
         oStack = permute(oStack, vnNewOrder);
      end

      % ctranspose - METHOD Overloaded transpose function
      function varargout = transpose(varargin) %#ok<STOUT>
         error('TensorStack:NotSupported', ...
               '*** TensorStack: ''transpose'' is not supported.');
      end

      % ctranspose - METHOD Overloaded ctranspose function
      function varargout = ctranspose(varargin) %#ok<STOUT>
         error('TensorStack:NotSupported', ...
               '*** TensorStack: ''ctranspose'' is not supported.');
      end

      %% -- Overloaded disp
      function disp(oStack)
         strSize = strtrim(sprintf('%dx', size(oStack)));
         strSize = strSize(1:end-1);

         disp(sprintf('   %s <a href="matlab:help TensorStack">TensorStack</a> object. <a href="matlab:methods(''TensorStack'')">Methods</a> <a href="matlab:properties(''TensorStack'')">Properties</a>', strSize)); %#ok<DSPS>
         fprintf(1, '\n');
      end
      
      %% -- Overloaded test functions
      function bIsNumeric = isnumeric(oStack)
         bIsNumeric = isnumeric(oStack.ctTensors{1});
      end
      
      function bIsReal = isreal(oStack)
         bIsReal = isreal(oStack.ctTensors{1});
      end
      
      function bIsLogical = islogical(oStack)
         bIsLogical = islogical(oStack.ctTensors{1});
      end
      
      function bIsScalar = isscalar(oStack)
         bIsScalar = numel(oStack) == 1;
      end
      
      function bIsMatrix = ismatrix(oStack)
         bIsMatrix = ~isscalar(oStack);
      end
      
      function bIsChar = ischar(oStack)
         bIsChar = ischar(oStack.ctTensors{1});
      end
      
      function bIsNan = isnan(oStack) %#ok<MANU>
         bIsNan = false;
      end
   end
   
   
end

%% -- Helper functions

% iscolon - FUNCTION Test whether a reference is equal to ':'
function bIsColon = iscolon(ref)
   bIsColon = ischar(ref) && isequal(ref, ':');
end

% isvalidsubscript - FUNCTION Test whether a vector contains valid entries
% for subscript referencing
function isvalidsubscript(oRefs)
   try
      % - Test for colon
      if (iscolon(oRefs))
         return;
      end
      
      if (islogical(oRefs))
         % - Test for logical indexing
         validateattributes(oRefs, {'logical'}, {'binary'});
         
      else
         % - Test for normal indexing
         validateattributes(oRefs, {'single', 'double'}, {'integer', 'real', 'positive'});
      end
      
   catch
      error('MappedTensor:badsubscript', ...
            '*** MappedTensor: Subscript indices must either be real positive integers or logicals.');
   end
end

function coCleanSubs = cleansubs(coSubs, vnTensorSize)
    % - Test for valid subscripts
    cellfun(@isvalidsubscript, coSubs);

    % - Get stack information
    nNumDims = numel(vnTensorSize);

    % - Check dimensionality
    if any(cellfun(@(c)c ~= 1, coSubs(nNumDims+1:end)))
       error('TensorStack:badsubscript', ...
             '*** TensorStack: Index exceds matrix dimensions.');
    end

    % - Trim trailing dimensions
    coCleanSubs = coSubs(1:min(numel(coSubs), nNumDims));

    % - Convert logical indices to linear indices
    for ii=1:numel(coCleanSubs)
       if islogical(coCleanSubs{ii})
           coCleanSubs{ii} = find(coCleanSubs{ii});
       end
    end

    % - Flatten indices arrays
    coCleanSubs = cellfun(@(c) c(:), coCleanSubs, 'un', false);
end