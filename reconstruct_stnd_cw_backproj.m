
function [fwd_mesh,pj_error] = reconstruct_stnd_cw_opt_fwd_exp(fwd_mesh,...
    data_fn,...
    iteration,...
    output_fn,...
    filter_n)

%
% CW Reconstruction program for standard meshes without regularization
%
% fwd_mesh is the input mesh (variable or filename)
% data_fn is the boundary data (variable or filename)
% iteration is the max number of iterations
% output_fn is the root output filename
% filter_n is the number of mean filters




% set modulation frequency to zero.
frequency = 0;

std = tic;

%****************************************
% If not a workspace variable, load mesh
if ischar(fwd_mesh)== 1
    fwd_mesh = load_mesh(fwd_mesh);
end
if ~strcmp(fwd_mesh.type,'stnd')
    errordlg('Mesh type is incorrect','NIRFAST Error');
    error('Mesh type is incorrect');
end

anom = load_data(data_fn);
if ~isfield(anom,'paa')
    errordlg('Data not found or not properly formatted','NIRFAST Error');
    error('Data not found or not properly formatted');
end
anom = anom.paa;
anom = log(anom(:,1));
% find NaN in data
datanum = 0;
[ns,junk]=size(fwd_mesh.source.coord);
for i = 1 : ns
    for j = 1 : length(fwd_mesh.link(i,:))
        datanum = datanum + 1;
        if fwd_mesh.link(i,j) == 0
            anom(datanum,:) = NaN;
        end
    end
end

ind = find(isnan(anom(:,1))==1);
% set mesh linkfile not to calculate NaN pairs:
link = fwd_mesh.link';
link(ind) = 0;
fwd_mesh.link = link';
clear link
% remove NaN from data
ind = setdiff(1:size(anom,1),ind);
anom = anom(ind,:);
clear ind;


% Initiate projection error
pj_error = [];

% Initiate log file
fid_log = fopen([output_fn '.log'],'w');
fprintf(fid_log,'Forward Mesh   = %s\n',fwd_mesh.name);
fprintf(fid_log,'Frequency      = %f MHz\n',frequency);
if ischar(data_fn) ~= 0
    fprintf(fid_log,'Data File      = %s\n',data_fn);
end
fprintf(fid_log,'Filter         = %d\n',filter_n);
fprintf(fid_log,'Output Files   = %s_mua.sol\n',output_fn);
fprintf(fid_log,'               = %s_mus.sol **CW recon only**\n',output_fn);


for it = 1 : iteration
    a = tic;
    % Calculate jacobian
    [J,data]=jacobian_stnd(fwd_mesh,frequency);
    
    % Set jacobian as Phase and Amplitude part instead of complex
    J = J.complete;
    
    % Read reference data
    clear ref;
    ref(:,1) = log(data.amplitude);
    
    data_diff = (anom-ref);
    
    pj_error = [pj_error sum(abs(data_diff.^2))];
    
    disp('---------------------------------');
    disp(['Iteration Number          = ' num2str(it)]);
    disp(['Projection error          = ' num2str(pj_error(end))]);
    
    fprintf(fid_log,'---------------------------------\n');
    fprintf(fid_log,'Iteration Number          = %d\n',it);
    fprintf(fid_log,'Projection error          = %f\n',pj_error(end));
    
    if it ~= 1
        p = (pj_error(end-1)-pj_error(end))*100/pj_error(end-1);
        disp(['Projection error change   = ' num2str(p) '%']);
        fprintf(fid_log,'Projection error change   = %f %%\n',p);
        if p <= 1
            disp('---------------------------------');
            disp('STOPPING CRITERIA REACHED');
            fprintf(fid_log,'---------------------------------\n');
            fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
            break
        end
    end
    
    
    % Normalize Jacobian wrt optical values
    N = fwd_mesh.mua;
    nn = length(fwd_mesh.nodes);
    % Normalise by looping through each node, rather than creating a
    % diagonal matrix and then multiplying - more efficient for large meshes
    for i = 1 : nn
        J(:,i) = J(:,i).*N(i,1);
    end
    clear nn N
    a1 = toc(a)
    a2 = tic;
    if it == 1
        %Initialize alpha
        alpha = 1;
    end
    
    % Finding Optimized alpha
    alpha = fminsearch(@(alpha) opt_mu_objfun(J, data_diff, alpha),alpha, optimset( 'MaxIter', 1000, 'TolX', 1e-16));
    
    foo = alpha*((J'*data_diff));
    
    clear J data_diff;
    
    foo = foo.*fwd_mesh.mua;
    a3 = toc(a2)
    %Update
    fwd_mesh.mua = fwd_mesh.mua + foo;
    
    clear foo
    clear J data_diff;
    
    % Filtering if needed!
    if filter_n > 1
        fwd_mesh = mean_filter(fwd_mesh,abs(filter_n));
    elseif filter_n < 0
        fwd_mesh = median_filter(fwd_mesh,abs(filter_n));
    end
    
    if it == 1
        fid = fopen([output_fn '_mua.sol'],'w');
    else
        fid = fopen([output_fn '_mua.sol'],'a');
    end
    fprintf(fid,'solution %g ',it);
    fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
    fprintf(fid,'-components=1 ');
    fprintf(fid,'-type=nodal\n');
    fprintf(fid,'%f ',fwd_mesh.mua);
    fprintf(fid,'\n');
    fclose(fid);
    
    if it == 1
        fid = fopen([output_fn '_mus.sol'],'w');
    else
        fid = fopen([output_fn '_mus.sol'],'a');
    end
    fprintf(fid,'solution %g ',it);
    fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
    fprintf(fid,'-components=1 ');
    fprintf(fid,'-type=nodal\n');
    fprintf(fid,'%f ',fwd_mesh.mus);
    fprintf(fid,'\n');
    fclose(fid);
    
end

% close log file!
time = toc(std);
fprintf(fid_log,'Computation TimeRegularization = %f\n',time);
fclose(fid_log);





function [recon_mesh] = interpolatef2r(fwd_mesh,recon_mesh)

% This function interpolates fwd_mesh into recon_mesh
NNC = size(recon_mesh.nodes,1);

for i = 1 : NNC
    if fwd_mesh.fine2coarse(i,1) ~= 0
        recon_mesh.mua(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
            fwd_mesh.mua(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
        recon_mesh.mus(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
            fwd_mesh.mus(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
        recon_mesh.kappa(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
            fwd_mesh.kappa(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
        recon_mesh.region(i,1) = ...
            median(fwd_mesh.region(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    elseif fwd_mesh.fine2coarse(i,1) == 0
        dist = distance(fwd_mesh.nodes,...
            fwd_mesh.bndvtx,...
            recon_mesh.nodes(i,:));
        mindist = find(dist==min(dist));
        mindist = mindist(1);
        recon_mesh.mua(i,1) = fwd_mesh.mua(mindist);
        recon_mesh.mus(i,1) = fwd_mesh.mus(mindist);
        recon_mesh.kappa(i,1) = fwd_mesh.kappa(mindist);
        recon_mesh.region(i,1) = fwd_mesh.region(mindist);
    end
end

function [fwd_mesh,recon_mesh] = interpolatep2f(fwd_mesh,recon_mesh)

for i = 1 : length(fwd_mesh.nodes)
    fwd_mesh.mua(i,1) = ...
        (recon_mesh.coarse2fine(i,2:end) * ...
        recon_mesh.mua(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
    fwd_mesh.kappa(i,1) = ...
        (recon_mesh.coarse2fine(i,2:end) * ...
        recon_mesh.kappa(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
    fwd_mesh.mus(i,1) = ...
        (recon_mesh.coarse2fine(i,2:end) * ...
        recon_mesh.mus(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
end
