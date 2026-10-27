import React from 'react';

interface AqiScaleProps {
  value: number;
  category?: string;
}

interface Band {
  name: string;
  range: string;
  min: number;
  max: number;
  color: string;
  textColor: string;
  bgLight: string;
  borderLight: string;
}

export const AQI_BANDS: Band[] = [
  { name: 'Good', range: '0–50', min: 0, max: 50, color: '#10b981', textColor: 'text-emerald-800', bgLight: 'bg-emerald-50', borderLight: 'border-emerald-200' },
  { name: 'Satisfactory', range: '51–100', min: 51, max: 100, color: '#84cc16', textColor: 'text-lime-800', bgLight: 'bg-lime-50', borderLight: 'border-lime-200' },
  { name: 'Moderate', range: '101–200', min: 101, max: 200, color: '#f59e0b', textColor: 'text-amber-800', bgLight: 'bg-amber-50', borderLight: 'border-amber-200' },
  { name: 'Poor', range: '201–300', min: 201, max: 300, color: '#f97316', textColor: 'text-orange-800', bgLight: 'bg-orange-50', borderLight: 'border-orange-200' },
  { name: 'Very Poor', range: '301–400', min: 301, max: 400, color: '#ef4444', textColor: 'text-red-800', bgLight: 'bg-red-50', borderLight: 'border-red-200' },
  { name: 'Severe', range: '401–500', min: 401, max: 500, color: '#881337', textColor: 'text-rose-900', bgLight: 'bg-rose-50', borderLight: 'border-rose-200' },
];

export const getCategoryStyles = (category: string) => {
  const norm = category.toLowerCase();
  switch (norm) {
    case 'good':
      return { bg: 'bg-emerald-500', text: 'text-emerald-800', bgLight: 'bg-emerald-50', border: 'border-emerald-200', badge: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
    case 'satisfactory':
      return { bg: 'bg-lime-500', text: 'text-lime-800', bgLight: 'bg-lime-50', border: 'border-lime-200', badge: 'bg-lime-100 text-lime-800 border-lime-300' };
    case 'moderate':
      return { bg: 'bg-amber-500', text: 'text-amber-800', bgLight: 'bg-amber-50', border: 'border-amber-200', badge: 'bg-amber-100 text-amber-800 border-amber-300' };
    case 'poor':
      return { bg: 'bg-orange-500', text: 'text-orange-800', bgLight: 'bg-orange-50', border: 'border-orange-200', badge: 'bg-orange-100 text-orange-800 border-orange-300' };
    case 'very poor':
      return { bg: 'bg-red-500', text: 'text-red-800', bgLight: 'bg-red-50', border: 'border-red-200', badge: 'bg-red-100 text-red-800 border-red-300' };
    case 'severe':
      return { bg: 'bg-rose-800', text: 'text-rose-900', bgLight: 'bg-rose-50', border: 'border-rose-200', badge: 'bg-rose-100 text-rose-900 border-rose-300' };
    default:
      return { bg: 'bg-slate-500', text: 'text-slate-800', bgLight: 'bg-slate-50', border: 'border-slate-200', badge: 'bg-slate-100 text-slate-800 border-slate-300' };
  }
};

export const AqiScale: React.FC<AqiScaleProps> = ({ value, category }) => {
  // Clamp marker position percentage between 0% and 100% on the 0-500 scale
  const percentage = Math.min(Math.max((value / 500) * 100, 0), 100);

  return (
    <div className="w-full space-y-3">
      {/* Horizontal Bar */}
      <div className="relative pt-6 pb-2">
        {/* Value Pointer Pin */}
        <div
          className="absolute -top-1 transition-all duration-500 transform -translate-x-1/2 flex flex-col items-center z-10"
          style={{ left: `${percentage}%` }}
        >
          <div className="bg-slate-900 text-white text-xs font-bold px-2 py-0.5 rounded shadow-md border border-slate-700 whitespace-nowrap">
            {value.toFixed(1)} {value > 500 && '(>500)'}
          </div>
          <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[6px] border-t-slate-900" />
        </div>

        {/* Bands bar */}
        <div className="h-4 w-full rounded-full overflow-hidden flex shadow-inner border border-slate-200">
          <div style={{ width: '10%' }} className="bg-emerald-500" title="Good (0-50)" />
          <div style={{ width: '10%' }} className="bg-lime-500" title="Satisfactory (51-100)" />
          <div style={{ width: '20%' }} className="bg-amber-500" title="Moderate (101-200)" />
          <div style={{ width: '20%' }} className="bg-orange-500" title="Poor (201-300)" />
          <div style={{ width: '20%' }} className="bg-red-500" title="Very Poor (301-400)" />
          <div style={{ width: '20%' }} className="bg-rose-800" title="Severe (401-500)" />
        </div>

        {/* Numbers under bar */}
        <div className="flex justify-between text-[11px] text-slate-400 font-medium px-0.5 mt-1">
          <span>0</span>
          <span>50</span>
          <span>100</span>
          <span>200</span>
          <span>300</span>
          <span>400</span>
          <span>500+</span>
        </div>
      </div>

      {/* Band legend pill cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5 text-xs">
        {AQI_BANDS.map((b) => {
          const isSelected = category && category.toLowerCase() === b.name.toLowerCase();
          return (
            <div
              key={b.name}
              className={`p-1.5 rounded-md border text-center transition-all ${
                isSelected
                  ? `${b.bgLight} ${b.borderLight} ring-2 ring-slate-900 font-semibold shadow-sm`
                  : 'bg-white border-slate-200 opacity-80'
              }`}
            >
              <div className="flex items-center justify-center gap-1">
                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: b.color }} />
                <span className={b.textColor}>{b.name}</span>
              </div>
              <span className="text-[10px] text-slate-500">{b.range}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
