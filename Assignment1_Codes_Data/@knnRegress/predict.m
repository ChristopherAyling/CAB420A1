    % Test function: predict on Xtest
    function Yte = predict(obj,Xte)
      [Ntr,Mtr] = size(obj.Xtrain);
      [Nte,Mte] = size(Xte);
      classes = unique(obj.Ytrain);
      Yte = repmat(obj.Ytrain(1), [Nte,1]);
      K = min(obj.K, Ntr);
      for i=1:Nte
        dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);
        [tmp,idx] = sort(dist);
        
        % Our code here
        kclosest_vals = [];
        for j=1:K
            kclosest_vals = [kclosest_vals, obj.Ytrain(idx(j))];
        end
        Yte(i) = sum(kclosest_vals)/K;
      end;
    end
