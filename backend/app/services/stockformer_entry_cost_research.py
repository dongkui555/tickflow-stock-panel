"""Offline entry-pool attribution using frozen ranks and ledger fee arithmetic."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from app.services import stockformer_ranking_research as ranking
from app.strategy import paper

COST_PARTS=('slippage_drag','proportional_commission_drag','stamp_tax_drag','minimum_commission_drag')
METRICS=('gross_industry','net_industry','gross_absolute','net_absolute','cost_total',*COST_PARTS,
    'net_without_minimum_industry','minimum_commission_extra_yuan','cash_outlay')
POLICY=dict(version='frozen_full_entry_cost_attribution_v1',model_fits=0,calibration_fits=0,
    rank_scope='existing_full_reference_percentiles_before_entry_and_labels',top_percentile=.8,bottom_percentile=.2,
    scopes=['necessary','full_entry','full_entry_market'],groups=['all','top20','bottom20'],
    cost_order=['slippage','rounded_proportional_commission','rounded_stamp_tax','minimum_commission_increment'],
    cost_counterfactual_is_attribution_only=True,bootstrap=ranking.POLICY,activation_allowed=False)


def cost_waterfall(qty, entry, exit_price, execution):
    """Return drag telescopes to the exact ledger net; no orders or fee changes."""
    values=[qty,entry,exit_price,*[execution[k] for k in ['commission_pct','stamp_tax_pct','slippage_bps']]]
    if (not all(math.isfinite(v) for v in values) or qty<=0 or qty%paper.LOT_SIZE!=0 or entry<=0 or exit_price<=0
            or execution['commission_pct']<0 or execution['stamp_tax_pct']<0
            or not 0<=execution['slippage_bps']<10000
            or execution.get('minimum_commission',paper.MIN_COMMISSION)!=paper.MIN_COMMISSION):
        raise ValueError('Invalid or unsupported frozen cost attribution input')
    buy=paper.apply_slippage(entry,'buy',execution['slippage_bps'])
    sell=paper.apply_slippage(exit_price,'sell',execution['slippage_bps'])
    paid,received=qty*buy,qty*sell
    c,t=execution['commission_pct'],execution['stamp_tax_pct']
    proportional_buy=round(paid*c,2)
    proportional_sell=round(received*c,2)
    proportional_sell_tax=round(received*c+received*t,2)
    buy_fee=paper.buy_fee(qty,buy,c)
    sell_fee=paper.sell_fee(qty,sell,c,t)
    raw=exit_price/entry-1
    slipped=received/paid-1
    commission_net=(received-proportional_sell-(paid+proportional_buy))/(paid+proportional_buy)
    no_minimum=(received-proportional_sell_tax-(paid+proportional_buy))/(paid+proportional_buy)
    actual=(received-sell_fee-(paid+buy_fee))/(paid+buy_fee)
    return dict(gross_absolute=raw,net_absolute=actual,cost_total=raw-actual,
        slippage_drag=raw-slipped,proportional_commission_drag=slipped-commission_net,
        stamp_tax_drag=commission_net-no_minimum,minimum_commission_drag=no_minimum-actual,
        net_without_minimum=no_minimum,minimum_commission_extra_yuan=
            buy_fee-proportional_buy+sell_fee-proportional_sell_tax,
        buy_fee_yuan=buy_fee,sell_fee_yuan=sell_fee,cash_outlay=paid+buy_fee)


def daily_groups(frame, calendar, scope):
    """Missing top labels never promote another stock; ranks remain unchanged."""
    if scope not in POLICY['scopes']:
        raise ValueError('Unknown fixed entry scope')
    ranking._keys(frame)
    dates=sorted(set(calendar))
    if not set(frame['date'])<=set(dates):
        raise ValueError('Diagnostic date missing from calendar')
    if frame.group_by('date').agg(pl.col('market_gate').n_unique().alias('states')).filter(pl.col('states')!=1).height:
        raise ValueError('Inconsistent known-day market state')
    if frame.filter(pl.col('_cost_verified')&~pl.all_horizontal([pl.col(c).is_finite() for c in METRICS]).fill_null(False)).height:
        raise ValueError('Verified label has missing cost components')
    predicate=pl.col('_necessary_entry_pool')
    if scope!='necessary':
        predicate=predicate&pl.col('_entry_rules_pass')
    if scope=='full_entry_market':
        predicate=predicate&pl.col('_market_gate_pass')
    selected=frame.filter(predicate)
    grid=pl.DataFrame({'date':dates},schema={'date':pl.Date}).join(
        frame.select('date','market_gate').unique(),on='date',how='left',validate='1:1')
    results=[]
    for group,mask in [('all',pl.lit(True)),('top20',pl.col('percentile')>=.8),('bottom20',pl.col('percentile')<=.2)]:
        sample=selected.filter(mask)
        valid=pl.col('_cost_verified').fill_null(False)
        aggregate=sample.group_by('date').agg(pl.len().alias('rows'),valid.sum().alias('labeled_rows'),
            *[pl.col(c).filter(valid).mean().alias(c) for c in METRICS])
        truth=sample.filter(valid)
        corr=truth.group_by('date').agg(pl.when((pl.len()>=ranking.POLICY['min_cross_section'])&
            ranking._varies('pred_ret_raw_5d')&ranking._varies('net_industry')).then(
                pl.corr('pred_ret_raw_5d','net_industry',method='spearman')).otherwise(None).alias('rank_ic'))
        results.append(grid.join(aggregate,on='date',how='left',validate='1:1').join(corr,on='date',how='left',validate='1:1')
            .with_columns(pl.col('rows','labeled_rows').fill_null(0),pl.lit(scope).alias('scope'),pl.lit(group).alias('group')))
    return pl.concat(results,how='vertical_relaxed').sort('group','date')


def summarize(daily, calendar):
    if daily['date'].n_unique()!=daily.height:
        raise ValueError('Summary requires a single scope and group with unique dates')
    dates=sorted(set(calendar))
    if not set(daily['date'])<=set(dates):
        raise ValueError('Diagnostic date missing from calendar')
    result=dict(rows=int(daily['rows'].sum()),labeled_rows=int(daily['labeled_rows'].sum()),
        calendar_days=len(dates),observed_days=daily.filter(pl.col('labeled_rows')>0).height,
        label_statistics_not_account_returns=True)
    mapping={d:i for i,d in enumerate(dates)}
    for column in [*METRICS,'rank_ic']:
        observations=dict(zip(daily['date'],daily[column],strict=True))
        values=np.asarray([observations.get(d) for d in dates],dtype=float)
        count=int(np.isfinite(values).sum())
        result[column]=dict(mean=ranking._mean(values),days=count,ci95=ranking._interval(values)
            if column in ['gross_industry','net_industry','rank_ic'] else None,
            positive_fraction=float((values[np.isfinite(values)]>0).mean()) if count else None)
    result['nonoverlapping_offsets']=[]
    for offset in range(ranking.POLICY['block_sessions']):
        part=daily.filter(pl.Series([mapping[d]%ranking.POLICY['block_sessions']==offset for d in daily['date']],dtype=pl.Boolean))
        result['nonoverlapping_offsets'].append(dict(offset=offset,days=part.filter(pl.col('labeled_rows')>0).height,
            gross_industry=ranking._mean(part['gross_industry'].to_numpy()),net_industry=ranking._mean(part['net_industry'].to_numpy())))
    return result
