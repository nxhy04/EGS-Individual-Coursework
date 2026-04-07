%% Case D: Lunar Base Energy System (PV + Battery, Off-Grid)
% Linear-program implementation of the Case D mathematical model
% with mass-minimised capacity-expansion extension.
%
% Objective:
%   Minimise launched mass = 5*xPV + 10*Emax + M*sum(ds)*dt
%

clear; clc; close all;

%% User options
csvFile = 'caseD_moon_base_hourly.csv';
enforceZeroShedding = true;
useCyclicEquality   = true; % E(T+1) = E(1)
plotResults         = true;
makeTradeoffCurve   = true;
runHeuristicCompare = true;

%% Parameters
eta_ch = 0.95; % charge efficiency
eta_dis = 0.95; % discharge efficiency
c_rate = 0.5; % [1/h]
init_soc_frac = 0.50; % E(1) = 0.5*Emax
pv_mass = 5; % [kg/kWp]
bat_mass = 10; % [kg/kWh]
dt = 1; % [h]
M_shed = 1e6; % very high penalty [kg/kWh-equivalent]
eps_flow = 1e-6; % tie-breaker penalty on charge/discharge
nightTol = 1e-9; % night threshold for critical hour detection

%% Load data
if ~isfile(csvFile)
    error('File not found: %s. Put the CSV in the same folder as this script.', csvFile);
end

Tdata = readtable(csvFile);
requiredCols = {'pv_profile_per_kwp','life_support_kw','thermal_kw','comms_kw', ...
                'lighting_kw','science_kw','galley_kw','total_demand_kw'};
for k = 1:numel(requiredCols)
    if ~ismember(requiredCols{k}, Tdata.Properties.VariableNames)
        error('Missing required column: %s', requiredCols{k});
    end
end

pvProf = Tdata.pv_profile_per_kwp(:); % [kW per kWp]
L_sub = [Tdata.life_support_kw(:), Tdata.thermal_kw(:), Tdata.comms_kw(:), ...
         Tdata.lighting_kw(:), Tdata.science_kw(:), Tdata.galley_kw(:)];
L_from_subsystems = sum(L_sub, 2); % [kW]
L = Tdata.total_demand_kw(:); % [kW]

% verify total demand matches subsystem sum closely
demandMismatch = max(abs(L - L_from_subsystems));
fprintf('Max |total_demand - sum(subsystems)| = %.6e kW\n', demandMismatch);

T = numel(L);

%% Solve main LP
sol = solve_caseD_lp(L, pvProf, eta_ch, eta_dis, c_rate, init_soc_frac, ...
    pv_mass, bat_mass, M_shed, eps_flow, dt, enforceZeroShedding, ...
    useCyclicEquality, []);

if sol.exitflag <= 0
    error('linprog did not converge to an optimal solution. Exitflag = %d', sol.exitflag);
end

%% Unpack solution
xPV  = sol.xPV;
Emax = sol.Emax;
s    = sol.s;
pch  = sol.pch;
pdis = sol.pdis;
E    = sol.E;
ds   = sol.ds;
fval = sol.fval;
output = sol.output;

%% Verification checks
balanceResidual = s + pdis + ds - L - pch;
storageResidual = E(2:end) - E(1:end-1) - eta_ch*pch*dt + (pdis*dt/eta_dis);

maxBalanceResidual = max(abs(balanceResidual));
maxStorageResidual = max(abs(storageResidual));
maxPVViolation = max(s - pvProf*xPV);
maxChargePowerViolation = max(pch - c_rate*Emax);
maxDischargePowerViolation = max(pdis - c_rate*Emax);
maxEnergyUpperViolation = max(E - Emax);
minEnergyState = min(E);
maxDischargeCausalityViolation = max(pdis - eta_dis*E(1:end-1)/dt);
maxChargeHeadroomViolation = max(eta_ch*pch*dt - (Emax - E(1:end-1)));
maxSimulChargeDischarge = max(min(pch, pdis));
hoursSimulChargeDischarge = nnz(min(pch, pdis) > 1e-8);
endSOCgap = E(end) - E(1);

[maxPdis, tPdis] = max(pdis);
[maxPch, tPch] = max(pch);
[maxE, tE] = max(E);

%% Identify critical hour (worst deficit during night)
pvAvailable = pvProf * xPV;
nightMask = pvProf <= nightTol;
rawDeficit = max(L - pvAvailable, 0);
rawDeficitNight = rawDeficit;
rawDeficitNight(~nightMask) = -inf;
[criticalDeficit_kW, criticalHour] = max(rawDeficitNight);
if isinf(criticalDeficit_kW)
    criticalDeficit_kW = NaN;
    criticalHour = NaN;
end
criticalSOC_kWh = E(criticalHour);
criticalLoad_kW = L(criticalHour);

%% Results
fprintf('\n=== OPTIMISATION RESULTS ===\n');
fprintf('Exit flag: %d\n', sol.exitflag);
fprintf('Iterations: %d\n', output.iterations);
fprintf('Objective mass = %.3f kg\n', fval);
fprintf('xPV  = %.3f kWp\n', xPV);
fprintf('Emax = %.3f kWh\n', Emax);
fprintf('Total PV energy used on bus = %.3f kWh\n', sum(s)*dt);
fprintf('Total battery charge energy = %.3f kWh\n', sum(pch)*dt);
fprintf('Total battery discharge energy = %.3f kWh\n', sum(pdis)*dt);
fprintf('Total load served            = %.3f kWh\n', sum(L - ds)*dt);
fprintf('Total load shed             = %.6f kWh\n', sum(ds)*dt);
fprintf('Final minus initial SOC     = %.6f kWh\n', endSOCgap);

fprintf('\n=== VERIFICATION CHECKS ===\n');
fprintf('Max |bus balance residual|               = %.6e kW\n', maxBalanceResidual);
fprintf('Max |storage residual|                   = %.6e kWh\n', maxStorageResidual);
fprintf('Max PV availability violation            = %.6e kW\n', maxPVViolation);
fprintf('Max charge power constraint slack               = %.6e kW\n', maxChargePowerViolation);
fprintf('Max discharge power constraint slack            = %.6e kW\n', maxDischargePowerViolation);
fprintf('Max energy upper-bound violation         = %.6e kWh\n', maxEnergyUpperViolation);
fprintf('Min energy state (should be >= 0)        = %.6e kWh\n', minEnergyState);
fprintf('Max discharge causality violation        = %.6e kW\n', maxDischargeCausalityViolation);
fprintf('Max charge headroom violation            = %.6e kWh\n', maxChargeHeadroomViolation);
fprintf('Max simultaneous charge/discharge        = %.6e kW\n', maxSimulChargeDischarge);
fprintf('Hours with simultaneous charge/discharge = %d\n', hoursSimulChargeDischarge);

fprintf('\n=== EXTRA DIAGNOSTICS ===\n');
fprintf('max(pdis) = %.6f kW at hour %d\n', maxPdis, tPdis);
fprintf('max(pch)  = %.6f kW at hour %d\n', maxPch, tPch);
fprintf('max(E)    = %.6f kWh at hour %d\n', maxE, tE);
fprintf('Critical night-time deficit = %.6f kW at hour %d\n', criticalDeficit_kW, criticalHour);
fprintf('Load at critical hour       = %.6f kW\n', criticalLoad_kW);
fprintf('Stored energy at that hour  = %.6f kWh\n', criticalSOC_kWh);

%% Output table
Results = table((1:T)', L, pvAvailable, s, pch, pdis, ds, balanceResidual, ...
    E(1:end-1), E(2:end), rawDeficit, nightMask, ...
    'VariableNames', {'hour','load_kw','pv_available_kw','pv_used_kw','charge_kw', ...
    'discharge_kw','shed_kw','balance_residual_kw','E_start_kwh','E_end_kwh', ...
    'raw_deficit_kw','is_night'});

summary = table(xPV, Emax, fval, sum(ds)*dt, sum(s)*dt, sum(pch)*dt, sum(pdis)*dt, ...
    maxBalanceResidual, maxStorageResidual, endSOCgap, maxPdis, maxPch, maxE, ...
    criticalHour, criticalDeficit_kW, criticalLoad_kW, criticalSOC_kWh, ...
    maxDischargeCausalityViolation, maxChargeHeadroomViolation, ...
    'VariableNames', {'xPV_kWp','Emax_kWh','objective_mass_kg','shed_kWh', ...
    'pv_used_kWh','charge_kWh','discharge_kWh','max_balance_residual', ...
    'max_storage_residual','endSOCgap_kWh','max_pdis_kW','max_pch_kW', ...
    'max_E_kWh','critical_hour','critical_deficit_kW','critical_load_kW', ...
    'critical_SOC_kWh','max_discharge_causality_violation', ...
    'max_charge_headroom_violation'});

writetable(Results, 'caseD_lp_dispatch_results_enhanced.csv');
writetable(summary, 'caseD_lp_summary_enhanced.csv');

%% Tradeoff sweep: fix PV and find minimum battery / resulting mass
tradeoff = table();
if makeTradeoffCurve
    pvGrid = linspace(0.75*xPV, 1.30*xPV, 21)';
    Ereq = nan(size(pvGrid));
    massTotal = nan(size(pvGrid));
    shedEnergy = nan(size(pvGrid));
    feasible = false(size(pvGrid));

    fprintf('\n=== PV / BATTERY TRADEOFF SWEEP ===\n');
    for i = 1:numel(pvGrid)
        sol_i = solve_caseD_lp(L, pvProf, eta_ch, eta_dis, c_rate, init_soc_frac, ...
            pv_mass, bat_mass, M_shed, eps_flow, dt, enforceZeroShedding, ...
            useCyclicEquality, pvGrid(i));
        if sol_i.exitflag > 0
            Ereq(i) = sol_i.Emax;
            massTotal(i) = pv_mass*pvGrid(i) + bat_mass*sol_i.Emax;
            shedEnergy(i) = sum(sol_i.ds)*dt;
            feasible(i) = shedEnergy(i) <= 1e-6;
            fprintf('PV = %9.3f kWp | Emax = %10.3f kWh | mass = %12.3f kg | shed = %.3e kWh\n', ...
                pvGrid(i), Ereq(i), massTotal(i), shedEnergy(i));
        else
            fprintf('PV = %9.3f kWp | infeasible / no optimum\n', pvGrid(i));
        end
    end

    tradeoff = table(pvGrid, Ereq, massTotal, shedEnergy, feasible, ...
        'VariableNames', {'pv_kWp','Emax_kWh','total_mass_kg','shed_kWh','feasible'});
    writetable(tradeoff, 'caseD_pv_battery_tradeoff.csv');
end

%% Extension: heuristic sizing and comparison with LP optimum
heuristic = struct();
heuristicTable = table();

if runHeuristicCompare
    fprintf('\n=== MASS-MINISED CAPACITY-EXPANSION EXTENSION ===\n');
    fprintf('Running heuristic sizing comparison...\n');

    pvGridHeur = linspace(0.60*xPV, 1.50*xPV, 61)';
    heurEmax = nan(size(pvGridHeur));
    heurMass = nan(size(pvGridHeur));
    heurFeasible = false(size(pvGridHeur));

    for i = 1:numel(pvGridHeur)
        xpv_i = pvGridHeur(i);
        [Eheur_i, sim_i] = heuristic_find_min_battery_for_fixed_pv( ...
    L, pvProf, xpv_i, eta_ch, eta_dis, c_rate, init_soc_frac, dt, false);

        if sim_i.feasible
            heurEmax(i) = Eheur_i;
            heurMass(i) = pv_mass*xpv_i + bat_mass*Eheur_i;
            heurFeasible(i) = true;
            fprintf('Heuristic PV = %9.3f kWp | Emax = %10.3f kWh | mass = %12.3f kg\n', ...
                xpv_i, Eheur_i, heurMass(i));
        else
            fprintf('Heuristic PV = %9.3f kWp | infeasible\n', xpv_i);
        end
    end

    if any(heurFeasible)
        [heurBestMass, idxBest] = min(heurMass(heurFeasible));
        idxFeas = find(heurFeasible);
        idxBest = idxFeas(idxBest);

        heuristic.xPV = pvGridHeur(idxBest);
        heuristic.Emax = heurEmax(idxBest);
        heuristic.mass = heurMass(idxBest);

        heuristic.sim = heuristic_forward_sim(L, pvProf, heuristic.xPV, heuristic.Emax, ...
    eta_ch, eta_dis, c_rate, init_soc_frac, dt, false);

        heuristic.sim = heuristic_forward_sim(L, pvProf, heuristic.xPV, heuristic.Emax, ...
    eta_ch, eta_dis, c_rate, init_soc_frac, dt, false);

        heur_ds   = heuristic.sim.ds;
        heur_E    = heuristic.sim.E;
        heur_s    = heuristic.sim.s;
        heur_pch  = heuristic.sim.pch;
        heur_pdis = heuristic.sim.pdis;
        heur_pvAvail = pvProf * heuristic.xPV;
        
        heur_totalShed = sum(heur_ds) * dt;
        heur_maxShed = max(heur_ds);
        heur_hoursShed = nnz(heur_ds > 1e-8);
        heur_endSOCgap = heur_E(end) - heur_E(1);
        
        heur_loadResidual = heur_s + heur_pdis + heur_ds - L;
        heur_storageResidual = heur_E(2:end) - heur_E(1:end-1) ...
            - eta_ch*heur_pch*dt + (heur_pdis*dt/eta_dis);
        heur_pvViolation = heur_s + heur_pch - heur_pvAvail;
        
        fprintf('\n=== HEURISTIC VERIFICATION ===\n');
        fprintf('Heuristic feasible flag              = %d\n', heuristic.sim.feasible);
        fprintf('Heuristic total load shed            = %.12f kWh\n', heur_totalShed);
        fprintf('Heuristic max hourly shed            = %.12e kW\n', heur_maxShed);
        fprintf('Heuristic hours with shedding        = %d\n', heur_hoursShed);
        fprintf('Heuristic final minus initial SOC    = %.12f kWh\n', heur_endSOCgap);
        fprintf('Heuristic max |load residual|        = %.12e kW\n', max(abs(heur_loadResidual)));
        fprintf('Heuristic max |storage residual|     = %.12e kWh\n', max(abs(heur_storageResidual)));
        fprintf('Heuristic max PV availability viol.  = %.12e kW\n', max(heur_pvViolation));
        fprintf('Heuristic most-negative PV slack     = %.12e kW\n', min(heur_pvViolation));

        fprintf('\nHeuristic best design:\n');
        fprintf('xPV  = %.3f kWp\n', heuristic.xPV);
        fprintf('Emax = %.3f kWh\n', heuristic.Emax);
        fprintf('Mass = %.3f kg\n', heuristic.mass);

        fprintf('\nComparison against LP optimum:\n');
        fprintf('LP optimum mass      = %.3f kg\n', fval);
        fprintf('Heuristic mass       = %.3f kg\n', heuristic.mass);
        fprintf('Absolute mass gap    = %.3f kg\n', heuristic.mass - fval);
        fprintf('Relative mass gap    = %.3f %%\n', 100*(heuristic.mass - fval)/fval);
        fprintf('LP xPV               = %.3f kWp\n', xPV);
        fprintf('Heuristic xPV        = %.3f kWp\n', heuristic.xPV);
        fprintf('LP Emax              = %.3f kWh\n', Emax);
        fprintf('Heuristic Emax       = %.3f kWh\n', heuristic.Emax);

        heuristicTable = table(pvGridHeur, heurEmax, heurMass, heurFeasible, ...
            'VariableNames', {'pv_kWp','heuristic_Emax_kWh','heuristic_mass_kg','feasible'});
        writetable(heuristicTable, 'caseD_heuristic_tradeoff.csv');

        compareTable = table( ...
            xPV, Emax, fval, ...
            heuristic.xPV, heuristic.Emax, heuristic.mass, ...
            heuristic.mass - fval, 100*(heuristic.mass - fval)/fval, ...
            'VariableNames', {'lp_xPV_kWp','lp_Emax_kWh','lp_mass_kg', ...
            'heur_xPV_kWp','heur_Emax_kWh','heur_mass_kg', ...
            'mass_gap_kg','mass_gap_percent'});
        writetable(compareTable, 'caseD_lp_vs_heuristic_comparison.csv');
    else
        warning('No feasible heuristic design found over the tested PV grid.');
    end
end

%% Plots
if plotResults
    t = 1:T;

    figure('Name', 'Case D - Dispatch', 'Color', 'w');
    plot(t, L, 'LineWidth', 1.2); hold on;
    plot(t, pvAvailable, 'LineWidth', 1.2);
    plot(t, s, 'LineWidth', 1.2);
    plot(t, pch, 'LineWidth', 1.2);
    plot(t, pdis, 'LineWidth', 1.2);
    xline(criticalHour, '--k', 'Critical hour', 'LabelVerticalAlignment', 'bottom', ...
        'LabelOrientation', 'horizontal');
    if any(ds > 0)
        plot(t, ds, 'LineWidth', 1.2);
        legend('Load','PV available','PV used','Charge','Discharge','Critical hour','Shed', 'Location', 'best');
    else
        legend('Load','PV available','PV used','Charge','Discharge','Critical hour', 'Location', 'best');
    end
    xlabel('Hour'); ylabel('Power [kW]');
    title('Hourly dispatch'); grid on;
    ymax = 1.05 * max([L; pvAvailable; s; pch; pdis; ds; 1]);
    ylim([0 ymax]);

    figure('Name', 'Case D - Battery Energy', 'Color', 'w');
    plot(1:(T+1), E, 'LineWidth', 1.4); hold on;
    xline(criticalHour, '--k', 'Critical hour', 'LabelVerticalAlignment', 'bottom', ...
        'LabelOrientation', 'horizontal');
    xlabel('Hour index'); ylabel('Stored energy [kWh]');
    title('Battery state of charge / stored energy');
    grid on;

    figure('Name', 'Case D - Daylight pattern', 'Color', 'w');
    plot(t, pvProf, 'LineWidth', 1.2);
    xlabel('Hour'); ylabel('PV profile [kW per kWp]');
    title('PV profile per installed kWp');
    grid on;

    figure('Name', 'Case D - Energy balance residual', 'Color', 'w');
    plot(t, balanceResidual, 'LineWidth', 1.2); hold on;
    yline(0, '--k');
    xlabel('Hour'); ylabel('Residual [kW]');
    title('Bus energy-balance residual: s + pdis + ds - L - pch');
    grid on;

    if makeTradeoffCurve && ~isempty(tradeoff)
        valid = isfinite(tradeoff.Emax_kWh) & tradeoff.feasible;
        figure('Name', 'Case D - Mass vs PV/Battery tradeoff', 'Color', 'w');
        yyaxis left
        plot(tradeoff.pv_kWp(valid), tradeoff.total_mass_kg(valid), '-o', 'LineWidth', 1.2);
        ylabel('Total launched mass [kg]');
        yyaxis right
        plot(tradeoff.pv_kWp(valid), tradeoff.Emax_kWh(valid), '-s', 'LineWidth', 1.2);
        ylabel('Required battery capacity [kWh]');
        xlabel('Fixed PV capacity [kWp]');
        title('Mass vs PV/battery tradeoff curve');
        grid on;
        hold on;
        yyaxis left
        plot(xPV, fval, 'p', 'MarkerSize', 12, 'LineWidth', 1.4);
        text(xPV, fval, '  optimum', 'VerticalAlignment', 'bottom');
        legend('Total mass','Optimum','Required battery capacity', 'Location', 'best');
    end

    if runHeuristicCompare && ~isempty(heuristicTable)
        validH = heuristicTable.feasible & isfinite(heuristicTable.heuristic_Emax_kWh);
        figure('Name', 'Case D - LP vs Heuristic Comparison', 'Color', 'w');
        yyaxis left
        plot(heuristicTable.pv_kWp(validH), heuristicTable.heuristic_mass_kg(validH), '-o', 'LineWidth', 1.2); hold on;
        yline(fval, '--', 'LP optimum mass');
        ylabel('Launched mass [kg]');

        yyaxis right
        plot(heuristicTable.pv_kWp(validH), heuristicTable.heuristic_Emax_kWh(validH), '-s', 'LineWidth', 1.2);
        ylabel('Heuristic battery capacity [kWh]');

        xlabel('PV capacity [kWp]');
        title('Mass-minimised capacity expansion: heuristic vs LP optimum');
        grid on;
    end
end

%% ========================= LOCAL FUNCTIONS ==============================

function sol = solve_caseD_lp(L, pvProf, eta_ch, eta_dis, c_rate, init_soc_frac, ...
    pv_mass, bat_mass, M_shed, eps_flow, dt, enforceZeroShedding, ...
    useCyclicEquality, fixed_xPV)

T = numel(L);

% Decision-variable indexing
ix_xPV  = 1;
ix_Emax = 2;
ix_s    = 3:(2 + T);
ix_pch  = (3 + T):(2 + 2*T);
ix_pdis = (3 + 2*T):(2 + 3*T);
ix_E    = (3 + 3*T):(3 + 4*T);
ix_ds   = (4 + 4*T):(3 + 5*T);
nvars   = 3 + 5*T;

% Objective vector
f = zeros(nvars, 1);
f(ix_xPV) = pv_mass;
f(ix_Emax) = bat_mass;
f(ix_ds) = M_shed * dt;
f(ix_pch) = eps_flow * dt;
f(ix_pdis) = eps_flow * dt;

% Bounds
lb = zeros(nvars, 1);
ub = inf(nvars, 1);
ub(ix_ds) = L;
if enforceZeroShedding
    ub(ix_ds) = 0;
end
if ~isempty(fixed_xPV)
    lb(ix_xPV) = fixed_xPV;
    ub(ix_xPV) = fixed_xPV;
end

% Equality constraints
nEq = T + T + 1 + double(useCyclicEquality);
Aeq = spalloc(nEq, nvars, 8*T + 4);
beq = zeros(nEq, 1);
row = 0;

for t = 1:T
    row = row + 1;
    Aeq(row, ix_s(t)) = 1;
    Aeq(row, ix_pdis(t)) = 1;
    Aeq(row, ix_ds(t)) = 1;
    Aeq(row, ix_pch(t)) = -1;
    beq(row) = L(t);
end

for t = 1:T
    row = row + 1;
    Aeq(row, ix_E(t+1)) = 1;
    Aeq(row, ix_E(t)) = -1;
    Aeq(row, ix_pch(t)) = -eta_ch * dt;
    Aeq(row, ix_pdis(t)) = (dt / eta_dis);
end

row = row + 1;
Aeq(row, ix_E(1)) = 1;
Aeq(row, ix_Emax) = -init_soc_frac;

if useCyclicEquality
    row = row + 1;
    Aeq(row, ix_E(T+1)) = 1;
    Aeq(row, ix_E(1)) = -1;
end

% Inequality constraints
nIneq = T + T + T + (T + 1) + T + T + double(~useCyclicEquality);
A = spalloc(nIneq, nvars, 10*T + 3);
b = zeros(nIneq, 1);
row = 0;

for t = 1:T
    row = row + 1;
    A(row, ix_s(t)) = 1;
    A(row, ix_xPV) = -pvProf(t);
end

for t = 1:T
    row = row + 1;
    A(row, ix_pch(t)) = 1;
    A(row, ix_Emax) = -c_rate;
end

for t = 1:T
    row = row + 1;
    A(row, ix_pdis(t)) = 1;
    A(row, ix_Emax) = -c_rate;
end

for t = 1:(T+1)
    row = row + 1;
    A(row, ix_E(t)) = 1;
    A(row, ix_Emax) = -1;
end

for t = 1:T
    row = row + 1;
    A(row, ix_pdis(t)) = 1;
    A(row, ix_E(t)) = -eta_dis / dt;
end

for t = 1:T
    row = row + 1;
    A(row, ix_pch(t)) = 1;
    A(row, ix_E(t)) = 1 / (eta_ch * dt);
    A(row, ix_Emax) = -1 / (eta_ch * dt);
end

if ~useCyclicEquality
    row = row + 1;
    A(row, ix_E(1)) = 1;
    A(row, ix_E(T+1)) = -1;
end

opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');
[z, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, opts);

if isempty(z)
    sol = struct('exitflag', exitflag, 'output', output, 'fval', NaN, ...
        'xPV', NaN, 'Emax', NaN, 's', nan(T,1), 'pch', nan(T,1), ...
        'pdis', nan(T,1), 'E', nan(T+1,1), 'ds', nan(T,1));
    return;
end

sol = struct();
sol.exitflag = exitflag;
sol.output = output;
sol.fval = fval;
sol.xPV  = z(ix_xPV);
sol.Emax = z(ix_Emax);
sol.s    = z(ix_s);
sol.pch  = z(ix_pch);
sol.pdis = z(ix_pdis);
sol.E    = z(ix_E);
sol.ds   = z(ix_ds);
end

function [Emin, simBest] = heuristic_find_min_battery_for_fixed_pv( ...
    L, pvProf, xPV, eta_ch, eta_dis, c_rate, init_soc_frac, dt, useCyclicEquality)
% For a fixed PV size, find the minimum battery capacity that yields
% feasible greedy operation over the horizon.

% Initial lower bound from peak hourly deficit / C-rate
pvAvail = pvProf * xPV;
deficit = max(L - pvAvail, 0);
Elo = max(deficit) / max(c_rate, 1e-12);

% Start with a reasonable upper bound and double until feasible
Ehi = max(sum(deficit)*dt, 1);  
simHi = heuristic_forward_sim(L, pvProf, xPV, Ehi, eta_ch, eta_dis, c_rate, ...
    init_soc_frac, dt, useCyclicEquality);

iter = 0;
while ~simHi.feasible
    Ehi = 2 * Ehi;
    simHi = heuristic_forward_sim(L, pvProf, xPV, Ehi, eta_ch, eta_dis, c_rate, ...
        init_soc_frac, dt, useCyclicEquality);
    iter = iter + 1;
    if Ehi > 1e7 || iter > 60
        Emin = NaN;
        simBest = struct('feasible', false);
        return;
    end
end

% Bisection on battery capacity
tol = 1e-3;
while (Ehi - Elo) > tol
    Emid = 0.5 * (Elo + Ehi);
    simMid = heuristic_forward_sim(L, pvProf, xPV, Emid, eta_ch, eta_dis, ...
        c_rate, init_soc_frac, dt, useCyclicEquality);

    if simMid.feasible
        Ehi = Emid;
        simHi = simMid;
    else
        Elo = Emid;
    end
end

Emin = Ehi;
simBest = simHi;
end

function sim = heuristic_forward_sim(L, pvProf, xPV, Emax, eta_ch, eta_dis, ...
    c_rate, init_soc_frac, dt, useCyclicEquality)
% Greedy heuristic dispatch:
% 1) use PV directly for load
% 2) charge battery with remaining PV
% 3) discharge battery to meet remaining deficit
% Feasible if no unmet load and SOC constraints respected.

T = numel(L);
pvAvail = pvProf * xPV;

E = zeros(T+1, 1);
E(1) = init_soc_frac * Emax;

s = zeros(T,1);
pch = zeros(T,1);
pdis = zeros(T,1);
ds = zeros(T,1);

feasible = true;

for t = 1:T
    % Direct PV to load
    s(t) = min(L(t), pvAvail(t));

    % Surplus PV available for charging
    surplusPV = pvAvail(t) - s(t);
    chargePowerMax = c_rate * Emax;
    chargeHeadroomPower = (Emax - E(t)) / max(eta_ch * dt, 1e-12);
    pch(t) = max(0, min([surplusPV, chargePowerMax, chargeHeadroomPower]));

    % Remaining load deficit after direct PV
    remainingLoad = L(t) - s(t);

    % Battery discharge to serve remaining deficit
    dischargePowerMax = c_rate * Emax;
    causalityPower = eta_dis * E(t) / max(dt, 1e-12);
    pdis(t) = max(0, min([remainingLoad, dischargePowerMax, causalityPower]));

    % Any unmet load becomes shedding
    ds(t) = max(0, remainingLoad - pdis(t));

    % State update
    E(t+1) = E(t) + eta_ch*pch(t)*dt - (pdis(t)*dt/eta_dis);

    % Feasibility checks
    if ds(t) > 1e-8 || E(t+1) < -1e-8 || E(t+1) > Emax + 1e-8
        feasible = false;
        % keep sim data, but can stop early
        E(t+1:end) = E(t+1);
        break;
    end
end

if feasible
    % For the heuristic comparator, use a weaker terminal condition:
    % end SOC must not be below initial SOC.
    if E(end) + 1e-6 < E(1)
        feasible = false;
    end
end

sim = struct();
sim.feasible = feasible;
sim.s = s;
sim.pch = pch;
sim.pdis = pdis;
sim.E = E;
sim.ds = ds;
end